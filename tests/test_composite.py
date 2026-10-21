import torch
from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import detector
from claudeknows.composite import CompositePaste, crop_pool, _overlap


def _samples(tmp_path):
    root = tmp_path / "d"
    folder = root / "own" / "together"
    folder.mkdir(parents=True)
    Image.new("RGB", (200, 150), (250, 10, 10)).save(folder / "v1_1.jpg")
    Image.new("RGB", (200, 150), (10, 10, 250)).save(folder / "v2_1.jpg")
    dd.save_boxes(folder, {
        "v1_1.jpg": {"by": "owlv2", "boxes": [{"label": "shoes", "box": [0.1, 0.1, 0.4, 0.5]}]},
        "v2_1.jpg": {"by": "owlv2", "boxes": [{"label": "keys", "box": [0.5, 0.5, 0.9, 0.9]}]}})
    return dd.scan_detection(root)[0]


def _target():
    return {"boxes": torch.tensor([[10.0, 10.0, 60.0, 70.0]]), "labels": torch.tensor([1])}


def test_pool_has_every_box_with_class_index(tmp_path):
    pool = crop_pool(_samples(tmp_path))
    assert sorted(label for _, label, _ in pool) == [2, 3]  # keys=2, shoes=3


def test_p_zero_changes_nothing(tmp_path):
    img = torch.rand(3, 150, 200)
    out, t = CompositePaste(crop_pool(_samples(tmp_path)), p=0.0)(img, _target())
    assert torch.equal(out, img) and len(t["labels"]) == 1


def test_pasted_boxes_are_inside_apart_and_labelled(tmp_path):
    pool = crop_pool(_samples(tmp_path))
    paste = CompositePaste(pool, p=1.0, max_paste=3, seed=1)
    added = 0
    for _ in range(40):
        img = torch.full((3, 150, 200), 0.5)
        out, t = paste(img, _target())
        boxes, labels = t["boxes"].tolist(), t["labels"].tolist()
        assert labels[0] == 1 and boxes[0] == [10.0, 10.0, 60.0, 70.0]  # original kept
        assert set(labels) <= {1, 2, 3}
        for b in boxes:
            assert 0 <= b[0] < b[2] <= 200 and 0 <= b[1] < b[3] <= 150
            assert (b[2] - b[0]) <= 0.4 * 200 + 1 and (b[3] - b[1]) <= 0.4 * 150 + 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert _overlap(boxes[i], boxes[j]) <= 0.1
        added += len(labels) - 1
    assert added > 0


def test_pasted_pixels_come_from_the_source_crop(tmp_path):
    paste = CompositePaste(crop_pool(_samples(tmp_path)), p=1.0, max_paste=1, seed=3)
    img = torch.zeros(3, 150, 200)
    out, t = paste(img, _target())
    x0, y0, x1, y1 = (int(v) for v in t["boxes"][-1])
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    red_or_blue = out[:, cy, cx]
    assert float(red_or_blue.max()) > 0.9  # centre of the patch is fully pasted (not feathered)


def test_training_with_composite_runs(tmp_path):
    samples = _samples(tmp_path)
    model, hist = detector.train_detector(samples, epochs=1, batch_size=2, pretrained=False,
                                          max_batches=1, device="cpu", composite=1.0,
                                          log=lambda *_: None)
    assert len(hist) == 1
