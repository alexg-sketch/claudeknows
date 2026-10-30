import random

import pytest
import torch
from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import detector


def test_zoom_crop_keeps_exact_shifted_box():
    img = torch.rand(3, 100, 200)
    target = {"boxes": torch.tensor([[90.0, 40.0, 110.0, 60.0]]), "labels": torch.tensor([1])}
    for seed in range(20):
        out_img, out = detector.zoom_crop(img, target, random.Random(seed))
        if out_img.shape == img.shape:
            continue  # no window fit this time
        assert 50 <= out_img.shape[-2] <= 70 and 100 <= out_img.shape[-1] <= 140
        w = out["boxes"][0, 2] - out["boxes"][0, 0]
        assert float(w) == pytest.approx(20.0) and len(out["labels"]) == 1
        # the pixels under the box are the original ones
        x0 = 90 - float(out["boxes"][0, 0])
        y0 = 40 - float(out["boxes"][0, 1])
        assert torch.equal(out_img, img[..., int(y0):int(y0) + out_img.shape[-2],
                                        int(x0):int(x0) + out_img.shape[-1]])
        return
    pytest.fail("zoom_crop never cropped")


def test_zoom_crop_skips_empty_and_unfittable():
    img = torch.rand(3, 100, 200)
    empty = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}
    assert detector.zoom_crop(img, empty, random.Random(0))[0] is img
    big = {"boxes": torch.tensor([[0.0, 0.0, 200.0, 100.0]]), "labels": torch.tensor([0])}
    assert detector.zoom_crop(img, big, random.Random(0))[0] is img  # box larger than any window


def _samples(tmp_path):
    root = tmp_path / "d"
    folder = root / "web" / "shoes"
    folder.mkdir(parents=True)
    boxes = {}
    for i in range(4):
        Image.new("RGB", (160, 120), (200, 50, 50)).save(folder / f"img{i}.jpg")
        boxes[f"img{i}.jpg"] = {"by": "openimages", "boxes": [
            {"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
    dd.save_boxes(folder, boxes)
    return dd.scan_detection(root)[0]


def test_training_with_zoom_and_ema_runs(tmp_path):
    samples = _samples(tmp_path)
    model, hist = detector.train_detector(samples, epochs=1, batch_size=2, pretrained=False,
                                          max_batches=2, device="cpu", log=lambda *_: None,
                                          zoom_crop_p=1.0, ema_decay=0.9, warmup=1)
    assert len(hist) == 1


def test_ema_weights_lie_between_and_decay_zero_is_plain(tmp_path):
    samples = _samples(tmp_path)
    kw = dict(epochs=1, batch_size=1, pretrained=False, max_batches=4, device="cpu",
              log=lambda *_: None, warmup=1)
    plain, _ = detector.train_detector(samples, **kw)
    off, _ = detector.train_detector(samples, ema_decay=0.0, **kw)
    ema, _ = detector.train_detector(samples, ema_decay=0.5, **kw)
    def gap(x, y):
        return max(float((v - y.state_dict()[k]).abs().max())
                   for k, v in x.state_dict().items() if v.dtype.is_floating_point)
    # two plain runs differ only by tiny run-to-run noise; the averaged weights differ far more
    assert gap(plain, off) < 0.1
    assert gap(plain, ema) > 5 * gap(plain, off)
