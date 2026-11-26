"""Task 29e: "phone camera" effects (motion blur, focus blur, noise, JPEG) on training photos."""
import random

import pytest
import torch
from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import detector, retrain

QUIET = lambda *_: None  # noqa: E731


def _target():
    return {"boxes": torch.tensor([[10.0, 12.0, 50.0, 40.0], [60.0, 5.0, 90.0, 55.0]]),
            "labels": torch.tensor([1, 3])}


def test_motion_blur_at_angle_zero_spreads_only_along_the_row():
    img = torch.zeros(3, 21, 21)
    img[:, 10, 10] = 1.0  # one white dot
    out = detector.motion_blur(img, 7, 0)
    rows = (out.sum(dim=(0, 2)) > 0).nonzero().flatten().tolist()
    assert rows == [10]
    assert int((out[0, 10] > 0).sum()) == 7  # spread over the blur length
    assert float(out.sum()) == pytest.approx(3.0, abs=1e-4)  # kernel sums to 1
    # a 1-px vertical line blurred left-right stays inside every row it was in
    line = torch.zeros(3, 21, 21)
    line[:, :, 10] = 1.0
    assert torch.allclose(detector.motion_blur(line, 5, 0)[:, 3:18, 8:13].sum(dim=-1),
                          torch.ones(3, 15), atol=1e-5)


def test_motion_blur_at_90_degrees_spreads_only_along_the_column():
    img = torch.zeros(3, 21, 21)
    img[:, 10, 10] = 1.0
    out = detector.motion_blur(img, 5, 90)
    assert (out.sum(dim=(0, 1)) > 0).nonzero().flatten().tolist() == [10]


@pytest.mark.parametrize("effect", ["motion blur", "focus blur", "noise", "compression"])
def test_every_effect_keeps_size_dtype_and_range(effect):
    img = torch.rand(3, 240, 320)  # blur length is 0.5-1.5% of the short side
    out, name = detector.degrade(img, random.Random(1), [effect])
    assert name == effect
    assert out.shape == img.shape and out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(out, img)


def test_jpeg_roundtrip_keeps_shape_float_and_colour_order():
    red = torch.zeros(3, 32, 48)
    red[0] = 1.0
    out = detector.jpeg_roundtrip(red, 40)
    assert out.shape == red.shape and out.dtype == torch.float32
    r, g, b = (float(out[c].mean()) for c in range(3))
    assert r > 0.9 and g < 0.1 and b < 0.1  # still red, not blue


def test_transform_keeps_boxes_and_labels_with_degrade():
    img = torch.rand(3, 64, 96)
    # no geometric changes, so boxes must come out identical
    plain = detector.train_transform(rot_degrees=0, rot90_p=0, vflip_p=0, seed=3)
    noisy = detector.train_transform(rot_degrees=0, rot90_p=0, vflip_p=0, seed=3,
                                     degrade_p=1.0, log=QUIET)
    torch.manual_seed(0)
    a_img, a = plain(img, _target())
    torch.manual_seed(0)
    b_img, b = noisy(img, _target())
    assert torch.equal(a["boxes"], b["boxes"]) and torch.equal(a["labels"], b["labels"])
    assert not torch.equal(a_img, b_img)
    assert float(b_img.min()) >= 0 and float(b_img.max()) <= 1


def test_degrade_p_zero_reproduces_todays_output_exactly():
    img = torch.rand(3, 64, 96)
    old = detector.train_transform(seed=5)
    new = detector.train_transform(seed=5, degrade_p=0.0)
    torch.manual_seed(1)
    a_img, a = old(img, _target())
    torch.manual_seed(1)
    b_img, b = new(img, _target())
    assert torch.equal(a_img, b_img) and torch.equal(a["boxes"], b["boxes"])
    assert len(old.get_state()) == len(new.get_state()) == 3


def test_same_seed_same_choices():
    picks = [[detector.degrade(torch.rand(3, 40, 40), r)[1] for _ in range(30)]
             for r in (random.Random(7), random.Random(7))]
    assert picks[0] == picks[1] and len(set(picks[0])) > 1
    img = torch.rand(3, 40, 60)
    a, _ = detector.degrade(img, random.Random(9), ["noise"])
    b, _ = detector.degrade(img, random.Random(9), ["noise"])
    assert torch.equal(a, b)


def test_get_state_then_set_state_repeats_the_same_images():
    t = detector.train_transform(rot_degrees=0, rot90_p=0, vflip_p=0, degrade_p=0.7, seed=2,
                                 log=QUIET)
    photos = [torch.rand(3, 48, 64) for _ in range(5)]
    state = t.get_state()

    def run():
        torch.manual_seed(4)
        return [t(p, _target())[0] for p in photos]
    first = run()
    t.set_state(state)
    second = run()
    assert all(torch.equal(x, y) for x, y in zip(first, second))


def test_old_state_without_degrade_entry_still_loads():
    t = detector.train_transform(degrade_p=0.3, log=QUIET)
    t.set_state(t.get_state()[:2])  # what resume files saved before task 29e hold


def test_reseed_changes_degrade_draws():
    t = detector.train_transform(rot_degrees=0, rot90_p=0, vflip_p=0, degrade_p=1.0, log=QUIET)
    img = torch.rand(3, 48, 64)
    t.reseed(1)
    torch.manual_seed(0)
    a = t(img, _target())[0]
    t.reseed(2)
    torch.manual_seed(0)
    b = t(img, _target())[0]
    t.reseed(1)
    torch.manual_seed(0)
    c = t(img, _target())[0]
    assert torch.equal(a, c) and not torch.equal(a, b)


def test_recipe_includes_degrade_only_when_on():
    base = dict(epochs=3)
    assert "degrade_p" not in retrain.recipe_of(base)  # saved noise floors stay valid
    assert retrain.recipe_of({**base, "degrade_p": 0.3})["degrade_p"] == 0.3
    assert retrain.recipe_of({**base, "degrade_p": 0.3}) != retrain.recipe_of(base)


def test_stress_image_is_fixed_and_blurred():
    pil = Image.new("RGB", (480, 320), (0, 0, 0))  # 1% of 320 px = a 3 px blur
    pil.putpixel((240, 160), (255, 255, 255))
    a, b = detector.stress_image(pil, 3), detector.stress_image(pil, 3)
    assert a.size == pil.size and a.tobytes() == b.tobytes()
    assert a.tobytes() != pil.tobytes()


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


def test_training_with_degrade_runs_and_stress_line(tmp_path):
    samples = _samples(tmp_path)
    lines = []
    model, hist = detector.train_detector(samples, epochs=1, batch_size=2, pretrained=False,
                                          max_batches=2, device="cpu", log=lines.append,
                                          degrade_p=1.0, warmup=1)
    assert len(hist) == 1 and any("phone-camera effects" in m for m in lines)
    sharp = detector.predict_samples(model, samples)
    stressed = detector.predict_samples(model, samples, stress=True)
    assert "stress" in detector.format_stress(sharp, stressed)


def test_degrade_decision_rule():
    helped, same, drop = (0.03, (0.01, 0.05), "helped"), (0.0, (-0.005, 0.01), "no clear difference"), \
        (-0.02, (-0.04, -0.001), "hurt")
    assert detector.degrade_decision(same, helped, None)[0] is True
    assert detector.degrade_decision(same, None, helped)[0] is True
    assert detector.degrade_decision(drop, helped, helped)[0] is False  # validation may drop
    assert detector.degrade_decision(same, same, same)[0] is False  # no clear gain
    assert detector.degrade_decision(None, helped, helped)[0] is False


def test_compare_checkpoints_prints_rows(tmp_path):
    samples = _samples(tmp_path)
    kw = dict(epochs=1, batch_size=2, pretrained=False, max_batches=1, device="cpu", log=QUIET, warmup=1)
    for name, p in (("a.pt", 0.0), ("b.pt", 1.0)):
        model, hist = detector.train_detector(samples, degrade_p=p, **kw)
        detector.save_detector(model, str(tmp_path / name), hist)
    lines = detector.compare_checkpoints(str(tmp_path / "a.pt"), str(tmp_path / "b.pt"),
                                         samples[:2], samples[2:])
    assert any("validation" in x for x in lines) and "--degrade-p" in lines[-1]
