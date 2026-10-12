import json

import pytest
import torch
from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import openimages
from claudeknows.objects import import_own
from tests.test_objects_data import make_photo, make_video

HEAD = ("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,"
        "IsTruncated,IsGroupOf,IsDepiction,IsInside\n")


def _jpeg(path, size=(400, 300)):
    Image.new("RGB", size, (90, 120, 30)).save(path)


@pytest.fixture
def oi(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "ImageID,Source,LabelName,Confidence\n"
        "aaa,v,/m/0118n_9r,1.0\n"
        "ddd,v,/m/03v3yw,1.0\n"
        "eee,v,/m/03lnq3,1.0\n")  # eee will 404
    boxes = tmp_path / "boxes.csv"
    boxes.write_text(
        HEAD
        + "aaa,x,/m/04dr76w,1,0.1,0.5,0.1,0.9,0,0,0,0,0\n"
        + "aaa,x,/m/04dr76w,1,0.6,0.9,0.2,0.8,0,0,0,0,0\n"   # second bottle kept
        + "aaa,x,/m/04dr76w,1,0.0,0.01,0.0,0.01,0,0,0,0,0\n"  # tiny, dropped
        + "s1,x,/m/09j5n,1,0.0,0.5,0.0,0.5,0,0,0,0,0\n"
        + "s1,x,/m/09j5n,1,0.5,1.0,0.5,1.0,0,0,0,0,0\n"
        + "s2,x,/m/09j5n,1,0.0,1.0,0.0,1.0,0,0,1,0,0\n")      # group -> dropped
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("aaa", "ddd", "s1"):
        _jpeg(imgs / f"{n}.jpg", (1200, 800))
    return labels, boxes, imgs.as_uri() + "/{}.jpg"


def test_find_candidates_returns_every_box(oi):
    labels, boxes, _ = oi
    c = openimages.find_candidates(10, labels_url=labels.as_uri(), boxes_url=boxes.as_uri(),
                                   log=lambda *_: None, all_boxes=True, min_box_side=0.03)
    assert c["water_bottle"]["aaa"] == [(0.1, 0.5, 0.1, 0.9), (0.6, 0.9, 0.2, 0.8)]
    assert len(c["shoes"]["s1"]) == 2 and "s2" not in c["shoes"]
    assert c["keys"]["ddd"] is None


def test_download_keeps_whole_photos_and_boxes(tmp_path, oi):
    labels, boxes, url = oi
    c = openimages.find_candidates(10, labels_url=labels.as_uri(), boxes_url=boxes.as_uri(),
                                   log=lambda *_: None, all_boxes=True, min_box_side=0.03)
    c = {o: {i: (b or []) for i, b in v.items()} for o, v in c.items()}
    out = tmp_path / "web"
    logs = []
    dd.download_object("water_bottle", c["water_bottle"], out / "water_bottle", 5,
                       image_url=url, log=logs.append)
    dd.download_object("keys", c["keys"], out / "keys", 5, image_url=url, log=logs.append)
    img = Image.open(out / "water_bottle" / "aaa.jpg")
    assert img.size == (dd.DETECT_SIDE, round(800 * dd.DETECT_SIDE / 1200))  # whole photo, not a crop
    entry = dd.load_boxes(out / "water_bottle")["aaa.jpg"]
    assert entry["by"] == "openimages" and len(entry["boxes"]) == 2
    keys = dd.load_boxes(out / "keys")
    assert list(keys) == ["ddd.jpg"] and keys["ddd.jpg"]["by"] == "pending"  # eee 404 skipped
    # resuming does not duplicate
    dd.download_object("water_bottle", c["water_bottle"], out / "water_bottle", 5,
                       image_url=url, log=logs.append)
    assert len(dd.load_boxes(out / "water_bottle")) == 1


def test_parse_distance():
    assert dd.parse_distance("keys_1.0m.jpg") == ("keys", 1.0)
    assert dd.parse_distance("Shoes_2.5m_b.JPG") == ("shoes", 2.5)
    assert dd.parse_distance("water_bottle-3m.png") == ("water_bottle", 3.0)
    assert dd.parse_distance("keys_far.jpg") is None
    assert dd.parse_distance("hat_1m.jpg") is None


def test_import_own_footage_together_and_distance(tmp_path):
    drive = tmp_path / "claudeknows_photos"
    (drive / "together").mkdir(parents=True)
    (drive / "distance").mkdir()
    (drive / "keys").mkdir()
    make_video(drive / "together" / "t1.mp4", size=(160, 120))
    make_video(drive / "keys" / "k1.mp4", size=(160, 120))
    make_photo(drive / "distance" / "keys_1.0m.jpg")
    make_photo(drive / "distance" / "shoes_2.5m.jpg")
    make_photo(drive / "distance" / "oops.jpg")
    root = tmp_path / "det"
    logs = []
    s = dd.import_own_detection(drive, root, backend="opencv", log=logs.append)
    assert s["distance"]["photos"] == 2 and s["distance"]["skipped"] == ["oops.jpg"]
    assert s["together"]["videos"] == 1 and s["keys"]["videos"] == 1
    assert s["shoes"]["videos"] == 0  # missing folder is fine
    samples, pending = dd.scan_detection(root)
    assert samples == [] and pending > 0  # nothing boxed yet
    allp, _ = dd.scan_detection(root, boxed_only=False)
    assert {x.folder for x in allp} == {"keys", "together"}
    dist = dd.load_distance_photos(root)
    assert [(o, m) for _, o, m in dist] == [("keys", 1.0), ("shoes", 2.5)]
    # distance photos are never training samples
    assert all("distance" not in str(x.path) for x in allp)
    # re-running adds nothing
    again = dd.import_own_detection(drive, root, backend="opencv", log=logs.append)
    assert again["keys"]["frames"] == 0


def test_import_with_no_drive_folder(tmp_path):
    dd.import_own_detection(tmp_path / "nope", tmp_path / "det", log=lambda *_: None)
    assert dd.scan_detection(tmp_path / "det") == ([], 0)
    assert dd.load_distance_photos(tmp_path / "det") == []


def test_dataset_split_by_video_and_target_format(tmp_path):
    root = tmp_path / "det"
    d = root / "own" / "together"
    d.mkdir(parents=True)
    entries = {}
    for v in range(4):
        for k in range(3):
            name = f"vid_v{v}__f{k:04d}.jpg"
            make_photo(d / name, size=(200, 100))
            entries[name] = {"by": "owlv2", "boxes": [
                {"label": "keys", "box": [0.1, 0.2, 0.3, 0.6]},
                {"label": "shoes", "box": [0.5, 0.5, 1.0, 1.0]}]}
    make_photo(d / "vid_v9__f0001.jpg")  # unboxed frame
    entries["vid_v9__f0001.jpg"] = {"by": "pending", "boxes": []}
    dd.save_boxes(d, entries)
    samples, pending = dd.scan_detection(root)
    assert len(samples) == 12 and pending == 1
    train, test = dd.split_detection(samples)
    assert {s.group for s in train}.isdisjoint({s.group for s in test}) and test
    ds = dd.DetectionDataset(train)
    x, t = ds[0]
    assert x.shape == (3, 100, 200) and x.max() <= 1
    assert t["labels"].tolist() == [2, 3]  # keys=2, shoes=3 (0 is background)
    assert t["boxes"][0].tolist() == pytest.approx([20, 20, 60, 60])
    batch = dd.collate([ds[0], ds[1]])
    assert len(batch[0]) == 2
    assert "together" in dd.describe(train, test, pending)


def test_detection_data_feeds_torchvision_detector(tmp_path):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    root = tmp_path / "det"
    d = root / "web" / "shoes"
    d.mkdir(parents=True)
    make_photo(d / "a.jpg", size=(160, 120))
    dd.save_boxes(d, {"a.jpg": {"by": "openimages", "boxes": [
        {"label": "shoes", "box": [0.1, 0.1, 0.6, 0.7]}]}})
    samples, _ = dd.scan_detection(root)
    imgs, targets = dd.collate([dd.DetectionDataset(samples)[0]])
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
                                                  num_classes=4)
    model.train()
    losses = model(list(imgs), list(targets))
    assert torch.isfinite(sum(losses.values()))
