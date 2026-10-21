import json
import pytest

import torch
from PIL import Image

from claudeknows import autobox, detector
from claudeknows import detection_data as dd

_real_build = detector.build_detector


def _folder(tmp_path, name="keys", n=3):
    d = tmp_path / "own" / name
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (160, 120), (i * 40, 90, 30)).save(d / f"v{i}_0001.jpg")
    dd.register_pending(d)
    return d


def fake_detect(image, labels):
    # confident keys box on the first image size, a weak one, and a shoe
    return [("keys", 0.9, (0.1, 0.1, 0.5, 0.5)), ("keys", 0.85, (0.12, 0.1, 0.5, 0.52)),
            ("keys", 0.1, (0.6, 0.6, 0.9, 0.9)), ("shoes", 0.8, (0.5, 0.5, 0.9, 0.9))]


def test_autobox_keeps_confident_boxes_of_expected_object(tmp_path):
    d = _folder(tmp_path)
    stats = autobox.autobox_folder(d, fake_detect, expect="keys", log=lambda *_: None)
    assert stats == {"boxed": 3, "rejected": 0, "boxes": 3}  # NMS merged the near-duplicate
    entry = dd.load_boxes(d)["v0_0001.jpg"]
    assert entry["by"] == "owlv2" and entry["boxes"][0]["label"] == "keys"


def test_together_keeps_all_objects_and_rejects_empty(tmp_path):
    d = _folder(tmp_path, "together", 2)
    autobox.autobox_folder(d, lambda im, l: fake_detect(im, l) if im.getpixel((0, 0))[0] else [],
                           log=lambda *_: None)
    data = dd.load_boxes(d)
    assert data["v0_0001.jpg"]["by"] == "rejected"
    assert {b["label"] for b in data["v1_0001.jpg"]["boxes"]} == {"keys", "shoes"}


def test_rejected_not_used_and_not_pending(tmp_path):
    root = tmp_path
    d = _folder(root, "keys", 2)
    autobox.autobox_folder(d, lambda im, l: [], log=lambda *_: None)
    samples, pending = dd.scan_detection(root)
    assert samples == [] and pending == 0


def test_contact_sheet(tmp_path):
    d = _folder(tmp_path)
    autobox.autobox_folder(d, fake_detect, expect="keys", log=lambda *_: None)
    sheet = autobox.contact_sheet(d, tmp_path / "s.jpg", thumb=100)
    assert (tmp_path / "s.jpg").exists() and sheet.size[0] == 600


def test_average_precision_perfect_and_wrong():
    s = type("S", (), {})()
    gt = [("keys", (0.1, 0.1, 0.5, 0.5))]
    assert detector.average_precision([(s, gt, [("keys", 0.9, (0.1, 0.1, 0.5, 0.5))])], "keys") == 1.0
    assert detector.average_precision([(s, gt, [("keys", 0.9, (0.6, 0.6, 0.9, 0.9))])], "keys") == 0.0
    assert detector.average_precision([(s, gt, [])], "shoes") is None


def test_size_subset_ignores_other_objects():
    s = type("S", (), {})()
    big, small = (0.0, 0.0, 0.6, 0.6), (0.7, 0.7, 0.75, 0.75)
    gt = [("keys", big), ("keys", small)]
    preds = [("keys", 0.9, big)]  # finds the big one, misses the small one
    near = lambda b, _: detector.size_bin(b) == "near (big)"
    far = lambda b, _: detector.size_bin(b) == "far (small)"
    assert detector.average_precision([(s, gt, preds)], "keys", near) == 1.0
    assert detector.average_precision([(s, gt, preds)], "keys", far) == 0.0


def test_augmentation_keeps_boxes_inside_image():
    img = torch.rand(3, 120, 160)
    target = {"boxes": torch.tensor([[40.0, 30.0, 100.0, 90.0]]), "labels": torch.tensor([2])}
    tf = detector.train_transform()
    for _ in range(20):
        out_img, out = tf(img, target)
        assert out_img.shape == img.shape
        assert len(out["boxes"]) == len(out["labels"])
        if len(out["boxes"]):
            b = out["boxes"]
            assert (b[:, 0] >= 0).all() and (b[:, 2] <= 160).all() and (b[:, 3] <= 120).all()


def test_train_and_report_end_to_end(tmp_path):
    root = tmp_path / "det"
    folder = root / "web" / "shoes"
    folder.mkdir(parents=True)
    boxes = {}
    for i in range(4):
        Image.new("RGB", (160, 120), (200, 50, 50)).save(folder / f"img{i}.jpg")
        boxes[f"img{i}.jpg"] = {"by": "openimages", "boxes": [
            {"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
    dd.save_boxes(folder, boxes)
    samples, _ = dd.scan_detection(root)
    model, hist = detector.train_detector(samples, epochs=1, batch_size=2, pretrained=False,
                                          max_batches=2, device="cpu", log=lambda *_: None)
    assert len(hist) == 1
    ck = tmp_path / "d.pt"
    detector.save_detector(model, ck, hist)
    loaded, classes = detector.load_detector(ck)
    assert classes == ["water_bottle", "keys", "shoes"]
    results = detector.predict_samples(loaded, samples[:2])
    report = detector.build_report(results)
    text = detector.format_report(report)
    assert "by size in photo" in text and "by position" in text and "overall" in text
    found = detector.detect(loaded, Image.open(samples[0].path), min_score=0.0)
    assert all(set(f) == {"label", "score", "box"} for f in found)


def test_lr_factor_warms_up_then_decays():
    f = [detector.lr_factor(i, 1000, 200) for i in range(1000)]
    assert f[0] == pytest.approx(0.005) and f[199] == pytest.approx(1.0)
    assert f[200] == pytest.approx(1.0, abs=0.01)
    assert all(a >= b for a, b in zip(f[200:], f[201:]))  # only goes down after warm-up
    assert f[-1] == pytest.approx(0.02, abs=0.01)
    assert detector.lr_factor(5, 10, 200) > 0  # tiny runs (smoke tests) still work


def _tiny_samples(tmp_path, n=4):
    root = tmp_path / "data"
    folder = root / "own" / "shoes"
    folder.mkdir(parents=True)
    boxes = {}
    for i in range(n):
        Image.new("RGB", (160, 120), (200, 50, 50)).save(folder / f"v{i}_1.jpg")
        boxes[f"v{i}_1.jpg"] = {"by": "openimages", "boxes": [
            {"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
    dd.save_boxes(folder, boxes)
    return dd.scan_detection(root)[0]


def test_validation_keeps_best_epoch_and_stops_early(tmp_path, monkeypatch):
    samples = _tiny_samples(tmp_path)
    scores = iter([0.2, 0.5, 0.4, 0.3, 0.1, 0.9])  # best is epoch 2; stop 2 epochs later
    monkeypatch.setattr(detector, "mean_ap", lambda model, s, min_score=0.05: next(scores))
    stats = {}
    model, hist = detector.train_detector(
        samples[:2], epochs=6, batch_size=2, pretrained=False, max_batches=1, device="cpu",
        log=lambda *_: None, val_samples=samples[2:], patience=2, stats=stats)
    assert stats["best_epoch"] == 2 and stats["epochs_run"] == 4
    assert stats["val_map"] == [0.2, 0.5, 0.4, 0.3] and len(hist) == 4


def test_cli_train_with_validation(tmp_path, monkeypatch):
    _tiny_samples(tmp_path, n=8)
    monkeypatch.setattr(detector, "build_detector",
                        lambda **kw: _real_build(pretrained=False, min_size=128, max_size=160))
    detector.main(["train", "--root", str(tmp_path / "data"), "--epochs", "1", "--max-batches", "1",
                   "--batch-size", "2", "--no-pretrained", "--out", str(tmp_path / "d.pt")])
    ck = torch.load(tmp_path / "d.pt")
    assert ck["train_stats"]["epochs_run"] == 1 and len(ck["train_stats"]["val_map"]) == 1
