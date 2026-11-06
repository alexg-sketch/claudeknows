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
        data[f"v{i}_f0.jpg"] = {"by": "owlv2", "boxes": [{"label": "shoes", "box": [0.2, 0.2, 0.7, 0.8]}]}
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
