import pytest
from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import detector, gold


def _make(root):
    d = root / "own" / "shoes"
    d.mkdir(parents=True)
    data = {}
    for i in range(4):
        Image.new("RGB", (160, 120), (40 * i, 90, 90)).save(d / f"v{i}_f0.jpg")
        data[f"v{i}_f0.jpg"] = {"by": "checked", "boxes": [{"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
    dd.save_boxes(d, data)
    return dd.scan_detection(root)[0]


def test_build_verify_and_frozen(tmp_path):
    root = tmp_path / "det"
    samples = _make(root)
    m = gold.build_gold(samples[:2], root, log=lambda *_: None)
    assert len(m["images"]) == 2 and gold.verify_gold(m, root) == []
    with pytest.raises(FileExistsError):
        gold.build_gold(samples[:1], root, log=lambda *_: None)
    gold.build_gold(samples[:1], root, force=True, log=lambda *_: None)
    assert len(gold.load_gold(root)["images"]) == 1
    # a changed or missing file is reported
    s = samples[0].path
    Image.new("RGB", (160, 120), (1, 2, 3)).save(s)
    assert any("changed" in p for p in gold.verify_gold(gold.load_gold(root), root))
    s.unlink()
    assert any("missing" in p for p in gold.verify_gold(gold.load_gold(root), root))


def test_training_refuses_gold_and_pool_drops_it(tmp_path):
    root = tmp_path / "det"
    samples = _make(root)
    m = gold.build_gold(samples[:1], root, log=lambda *_: None)
    with pytest.raises(gold.GoldLeak):
        gold.assert_no_gold(samples, m)
    with pytest.raises(gold.GoldLeak):
        detector.train_detector(samples, epochs=1, batch_size=2, pretrained=False, max_batches=1,
                                device="cpu", log=lambda *_: None, gold=m)
    rest = gold.remove_gold(samples, m, log=lambda *_: None)
    assert len(rest) == 3
    gold.assert_no_gold(rest, m)
    detector.train_detector(rest, epochs=1, batch_size=2, pretrained=False, max_batches=1,
                            device="cpu", log=lambda *_: None, gold=m)
    # a byte-identical copy under another folder is still recognised
    copy = root / "own" / "copy.jpg"
    copy.write_bytes(samples[0].path.read_bytes())
    dup = dd.DetSample(copy, 0, "own", "copy", "shoes", (), "owlv2")
    assert gold.is_gold(dup, m)


def test_gold_samples_round_trip(tmp_path):
    root = tmp_path / "det"
    samples = _make(root)
    m = gold.build_gold(samples, root, log=lambda *_: None)
    back = gold.gold_samples(m, root)
    assert [b.boxes for b in back] == [s.boxes for s in samples]
    assert [b.path.resolve() for b in back] == [s.path.resolve() for s in samples]


def test_whole_gold_video_stays_out_of_training(tmp_path):
    root = tmp_path / "det"
    d = root / "own" / "keys"
    d.mkdir(parents=True)
    data = {}
    for f in range(3):  # three frames of ONE video, different bytes
        Image.new("RGB", (160, 120), (10 + 50 * f, 30, 30)).save(d / f"vid_kitchen__f{f}.jpg")
        data[f"vid_kitchen__f{f}.jpg"] = {"by": "checked", "boxes": [{"label": "keys", "box": [0.2, 0.2, 0.6, 0.7]}]}
    dd.save_boxes(d, data)
    samples = dd.scan_detection(root)[0]
    assert len({s.group for s in samples}) == 1
    m = gold.build_gold(samples[:1], root, log=lambda *_: None)
    # the other two frames are not byte-identical, but come from the gold video
    assert gold.remove_gold(samples, m, log=lambda *_: None) == []
    with pytest.raises(gold.GoldLeak):
        gold.assert_no_gold(samples[1:], m)


def test_unchecked_boxes_refused_and_confirmation_marks_checked(tmp_path):
    root = tmp_path / "det"
    d = root / "own" / "shoes"
    d.mkdir(parents=True)
    Image.new("RGB", (160, 120)).save(d / "v0_f0.jpg")
    Image.new("RGB", (160, 120)).save(d / "v1_f0.jpg")
    dd.save_boxes(d, {k: {"by": "owlv2", "boxes": [{"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
                      for k in ("v0_f0.jpg", "v1_f0.jpg")})
    cands = gold.candidates(dd.scan_detection(root)[0])
    assert len(cands) == 2
    with pytest.raises(ValueError):
        gold.build_gold(cands, root, log=lambda *_: None)
    ok = gold.mark_checked(cands, ["v1_f0.jpg"])
    assert [s.path.name for s in ok] == ["v1_f0.jpg"] and ok[0].by == "checked"
    m = gold.build_gold(ok, root, log=lambda *_: None)
    assert [e["by"] for e in m["images"]] == ["checked"]
    assert gold.review_sheet(cands, tmp_path / "s.jpg").exists()


def test_copy_to_drive_survives_wipe_and_is_not_overwritten(tmp_path):
    root = tmp_path / "det"
    samples = _make(root)
    m = gold.build_gold(samples[:2], root, log=lambda *_: None)
    drive = tmp_path / "drive" / "gold"
    gold.copy_gold(m, root, drive, log=lambda *_: None)
    import shutil
    shutil.rmtree(root)  # Colab session ended
    m2 = gold.load_gold(drive)
    assert gold.verify_gold(m2, drive) == [] and len(gold.gold_samples(m2, drive)) == 2
    gold.copy_gold(m2, drive, drive, log=lambda *_: None)  # idempotent
