import json

import numpy as np
import pytest
from PIL import Image

from claudeknows import distance as D


def _exif_jpeg(path, f35=26, size=(400, 300)):
    img = Image.new("RGB", size, (90, 90, 90))
    ex = Image.Exif()
    ex.get_ifd(D.EXIF_IFD)[D.TAG_FOCAL_35MM] = f35
    img.save(path, exif=ex)
    return path


def test_focal_from_exif_and_missing(tmp_path):
    p = _exif_jpeg(tmp_path / "a.jpg", 36, (400, 300))
    assert abs(D.focal_px_from_exif(Image.open(p)) - 400) < 1e-6
    Image.new("RGB", (10, 10)).save(tmp_path / "b.jpg")
    assert D.focal_px_from_exif(Image.open(tmp_path / "b.jpg")) is None


def test_size_prior_is_inverse_to_box_size():
    near = D.size_prior_metres("shoes", (0.0, 0.0, 0.5, 0.2), (400, 300), 400)
    far = D.size_prior_metres("shoes", (0.0, 0.0, 0.25, 0.1), (400, 300), 400)
    assert abs(near - 400 * 0.28 / 200) < 1e-9 and abs(far - 2 * near) < 1e-9
    assert D.size_prior_metres("shoes", (0, 0, 1, 1), (10, 10), None) is None


def test_box_value_uses_the_box_middle():
    m = np.zeros((100, 100)); m[40:60, 40:60] = 5.0
    assert D.box_value(m, (0.3, 0.3, 0.7, 0.7)) == 5.0




def test_estimate_orders_near_to_far_and_flags_disagreement():
    img = Image.new("RGB", (100, 100))

    def rel(im):
        m = np.zeros((100, 100)); m[:, :50] = 1.0; m[:, 50:] = 4.0
        return m

    def met(im):
        m = np.full((100, 100), 3.0); m[:, 50:] = 1.0
        return m
    dets = [{"label": "water_bottle", "score": 0.9, "box": (0.1, 0.1, 0.3, 0.4)},
            {"label": "keys", "score": 0.8, "box": (0.6, 0.6, 0.8, 0.8)}]
    res = D.estimate_distances(img, dets, rel, met, focal_px=100)
    assert [r["label"] for r in res] == ["keys", "water_bottle"] and res[0]["rank"] == 1
    assert res[0]["confidence"] == "low"
    bottle = res[1]
    assert bottle["metres_depth"] == 3.0 and bottle["metres_size"] == 100 * 0.22 / 30
    assert bottle["disagree"] and bottle["confidence"] == "low"
    lines = D.describe(res)
    assert "disagree" in lines[1] and lines[0].startswith("1. keys")
    # no EXIF: depth alone, no disagreement
    assert D.estimate_distances(img, dets[:1], rel, met)[0]["metres"] == 3.0
    assert D.estimate_distances(img, [], rel, met) == []


def test_check_distances_leave_one_out(tmp_path):
    root = tmp_path
    photos = []
    for i, true in enumerate([1.0, 2.0, 3.0, 4.0]):
        p = root / f"keys_{true}m.jpg"
        Image.new("RGB", (100, 100), (i, i, i)).save(p)
        photos.append((p, "keys", true))
    photos.append((root / "missing.jpg", "keys", 9.0))
    Image.new("RGB", (100, 100), (200, 0, 0)).save(root / "missing.jpg")

    def det(im):  # nothing found in the red photo
        if im.getpixel((0, 0))[0] == 200:
            return []
        return [{"label": "keys", "score": 0.9, "box": (0.4, 0.4, 0.6, 0.6)}]

    def met(im):  # model reads exactly half the truth (scale error); the photo index is the pixel
        true = [1.0, 2.0, 3.0, 4.0][im.getpixel((0, 0))[0]]
        return np.full((100, 100), true / 2)
    res = D.check_distances(photos, det, lambda im: np.ones((100, 100)), met)
    assert res["summary"]["found"] == 4 and len(res["missed"]) == 1
    d = res["summary"]["depth"]
    assert abs(d["median_error_pct"] - 50) < 1e-6
    assert d["median_error_pct_after_scale_fix"] < 1e-6  # constant scale error is fixed
    assert res["summary"]["order"]["depth"] == {"pairs": 6, "correct": 6}
    text = D.format_check(res)
    assert "50%" in text and "approximate" in text


def test_draw_results_marks_the_image():
    img = Image.new("RGB", (100, 80), (255, 255, 255))
    res = [{"label": "keys", "score": 0.9, "box": (0.1, 0.1, 0.6, 0.6), "metres": 1.5, "rank": 1}]
    out = D.draw_results(img, res)
    assert out.size == img.size and out.getpixel((10, 20)) != (255, 255, 255)
    assert img.getpixel((10, 20)) == (255, 255, 255)  # original untouched


def test_open_photo_reads_heic_with_camera_info(tmp_path):
    pytest.importorskip("pillow_heif")
    D._register_heif()
    img = Image.new("RGB", (400, 300), (90, 90, 90))
    ex = Image.Exif()
    ex.get_ifd(D.EXIF_IFD)[D.TAG_FOCAL_35MM] = 26
    img.save(tmp_path / "IMG_0001.HEIC", exif=ex)
    image, focal = D.open_photo(tmp_path / "IMG_0001.HEIC")
    assert image.mode == "RGB" and image.size == (400, 300)
    assert abs(focal - 26 / 36 * 400) < 1e-6


def test_open_photo_turns_phone_photos_upright(tmp_path):
    img = Image.new("RGB", (400, 300), (90, 90, 90))
    ex = Image.Exif()
    ex[0x0112] = 6  # Orientation: rotate 90 (phone held upright)
    ex.get_ifd(D.EXIF_IFD)[D.TAG_FOCAL_35MM] = 26
    img.save(tmp_path / "p.jpg", exif=ex)
    image, focal = D.open_photo(tmp_path / "p.jpg")
    assert image.size == (300, 400)
    assert abs(focal - 26 / 36 * 400) < 1e-6  # from the sensor's long side, before rotating


def test_open_photo_explains_unreadable_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a photo")
    with pytest.raises(ValueError, match="Could not open notes.txt"):
        D.open_photo(tmp_path / "notes.txt")
