import random
from types import SimpleNamespace

import pytest
import torch

from claudeknows import detector, retrain
from tests.test_zoom_ema import _samples

CLASSES = ["keys", "shoes"]
HIT, MISS = (0.0, 0.0, 0.5, 0.5), (0.6, 0.6, 1.0, 1.0)


def _results(n_groups, rates, seed=1):
    """Fake gold results: per group one box per object, found with probability rates[object]."""
    rng = random.Random(seed)
    out = []
    for g in range(n_groups):
        s = SimpleNamespace(source="own", group=f"v{g}", path=f"v{g}.jpg")
        gt = [(c, HIT) for c in CLASSES]
        preds = [(c, 0.9, HIT if rng.random() < rates[c] else MISS) for c in CLASSES]
        out.append((s, gt, preds))
    return out


def _gate(tmp_path):
    return retrain.GoldGate(tmp_path, CLASSES, n=200)


def test_identical_models_not_promoted(tmp_path):
    r = _results(30, {"keys": 0.5, "shoes": 0.5})
    d = _gate(tmp_path).check("cand", r, r, n_scenes=10, floor=0.0)
    assert not d["promote"] and any("not clearly better" in x for x in d["reasons"])


def test_better_overall_but_worse_on_one_object_not_promoted(tmp_path):
    cur = _results(40, {"keys": 0.0, "shoes": 1.0}, seed=2)
    cand = _results(40, {"keys": 1.0, "shoes": 0.7}, seed=2)
    d = _gate(tmp_path).check("cand", cur, cand, n_scenes=10, floor=0.0)
    assert d["overall"][1] > 0  # clearly better overall ...
    assert not d["promote"] and any(x.startswith("shoes may have got worse") for x in d["reasons"])


def test_gain_below_noise_floor_not_promoted_and_clear_gain_promoted(tmp_path):
    cur = _results(40, {"keys": 0.2, "shoes": 0.8}, seed=3)
    cand = _results(40, {"keys": 0.9, "shoes": 0.8}, seed=3)
    diffs = detector.paired_differences(cur, cand, CLASSES, n=200)
    assert diffs["overall"][1] > 0
    gain = diffs["overall"][0]
    low = retrain.decide(diffs, 10, floor=gain + 0.01, classes=CLASSES)
    assert not low["promote"] and any("noise" in x for x in low["reasons"])
    ok = retrain.decide(diffs, 10, floor=0.01, classes=CLASSES)
    assert ok["promote"]
    msg = retrain.promotion_message(ok)
    assert msg.startswith(f"promoted: at least {100 * diffs['overall'][1]:+.1f}")
    assert f"middle estimate {100 * gain:+.1f}" in msg
    # no floor measured yet, or too few gold scenes: never promoted
    assert not retrain.decide(diffs, 10, floor=None, classes=CLASSES)["promote"]
    few = retrain.decide(diffs, 4, floor=0.01, classes=CLASSES)
    assert not few["promote"] and few["reasons"][0].startswith("not enough evidence")


def test_second_candidate_refused_and_every_check_logged(tmp_path):
    r = _results(10, {"keys": 0.5, "shoes": 0.5})
    gate = _gate(tmp_path)
    gate.check("a", r, r, 10, 0.0)
    with pytest.raises(retrain.OneCandidateOnly):
        gate.check("b", r, r, 10, 0.0)
    rows = retrain.load_gold_log(tmp_path)
    assert [x["kind"] for x in rows] == ["gate"] and rows[0]["candidate"] == "a"
    retrain.record_noise_floor(tmp_path, [0.5, 0.52, 0.49], {"epochs": 1, "seeds": [0, 1, 2]})
    rows = retrain.load_gold_log(tmp_path)
    assert [x["kind"] for x in rows] == ["gate"] + ["noise-floor"] * 3
    assert retrain.load_noise_floor(tmp_path) == pytest.approx(0.03)
    assert retrain.load_noise_floor(tmp_path, {"epochs": 1}) == pytest.approx(0.03)
    assert retrain.load_noise_floor(tmp_path, {"epochs": 2}, log=lambda *_: None) is None


def test_wear_warning_at_budget_not_before(tmp_path):
    for i in range(19):
        retrain.log_gold_check(tmp_path, {"kind": "gate", "candidate": str(i)})
    retrain.log_gold_check(tmp_path, {"kind": "noise-floor", "candidate": "seed 0"})
    lines = retrain.gold_use_lines(tmp_path, "2026-11-09")
    assert lines == ["gold set used 20 times since 2026-11-09 (19 promotion checks, 1 noise-floor runs)"]
    retrain.log_gold_check(tmp_path, {"kind": "gate", "candidate": "19"})
    lines = retrain.gold_use_lines(tmp_path)
    assert len(lines) == 2 and lines[1].startswith("WARNING: the gold set has been used for 20+")


def _start(tmp_path, init_state, seed=0):
    """The weights a candidate starts from (no training steps: max_batches=0)."""
    m, _ = detector.train_detector(_samples(tmp_path), epochs=1, batch_size=2, pretrained=False,
                                   max_batches=0, device="cpu", log=lambda *_: None, seed=seed,
                                   init_state=init_state)
    return m.state_dict()


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_fresh_start_vs_current_start(tmp_path):
    torch.manual_seed(5)
    current = detector.build_detector(pretrained=False)
    detector.save_detector(current, tmp_path / "detector.pt")
    torch.manual_seed(0)
    fresh = detector.build_detector(pretrained=False).state_dict()
    coco = _start(tmp_path / "a", retrain.start_weights("coco", tmp_path / "detector.pt"))
    assert _same(coco, fresh) and not _same(coco, current.state_dict())
    cur = _start(tmp_path / "b", retrain.start_weights("current", tmp_path / "detector.pt"))
    assert _same(cur, current.state_dict())
    with pytest.raises(ValueError):
        retrain.start_weights("current", tmp_path / "missing.pt")


def test_versions_promote_and_one_line_rollback(tmp_path):
    torch.manual_seed(0)
    m = detector.build_detector(pretrained=False)
    detector.save_detector(m, tmp_path / "detector.pt")  # from before versioning
    from datetime import date
    a = retrain.save_version(m, tmp_path, extra={"init": "coco"}, today=date(2026, 11, 13))
    b = retrain.save_version(m, tmp_path, extra={"init": "coco"}, today=date(2026, 11, 13))
    assert (a, b) == ("detector_2026-11-13.pt", "detector_2026-11-13_2.pt")
    assert torch.load(tmp_path / a)["init"] == "coco"
    retrain.promote(tmp_path, a, log=lambda *_: None)
    retrain.promote(tmp_path, b, log=lambda *_: None)
    v = retrain.load_versions(tmp_path)
    assert v["promoted"] == [retrain.ORIGINAL, a, b] and v["current"] == b
    assert retrain.rollback(tmp_path, log=lambda *_: None) == a
    assert retrain.rollback(tmp_path, log=lambda *_: None) == retrain.ORIGINAL
    assert retrain.rollback(tmp_path, log=lambda *_: None) is None
    assert (tmp_path / "detector.pt").read_bytes() == (tmp_path / retrain.ORIGINAL).read_bytes()


def test_mix_keeps_original_photos_in_the_majority():
    base, extra = list(range(70)), [f"s{i}" for i in range(100)]
    mixed = retrain.mix_training(base, extra, log=lambda *_: None)
    n_self = sum(isinstance(x, str) for x in mixed)
    assert n_self == 30 and len(mixed) == 100
    with pytest.raises(ValueError):
        retrain.mix_training(base, extra, max_share=0.6, log=lambda *_: None)


def test_retrain_end_to_end_tiny(tmp_path):
    base = _samples(tmp_path / "train")
    models = tmp_path / "models"
    kw = dict(epochs=1, batch_size=2, max_batches=1, device="cpu")
    quiet = lambda *_: None
    first = retrain.retrain(base, [], [], models, pretrained=False, val_frac=0, train_kwargs=kw,
                            log=quiet)
    assert first["promoted"] and (models / "detector.pt").exists()
    # second run: a gold set with too few scenes, so the candidate is saved but not promoted
    gold = _samples(tmp_path / "gold")[:2]
    second = retrain.retrain(base, base[:1], gold, models, pretrained=False, val_frac=0,
                             train_kwargs=kw, gate=retrain.GoldGate(models, n=50), log=quiet)
    assert not second["promoted"]
    assert any("not enough evidence" in line for line in second["summary"])
    assert "1 self-labelled" in second["summary"][0]
    assert retrain.load_versions(models)["current"] == first["name"]
    assert [r["candidate"] for r in retrain.load_gold_log(models)] == [second["name"]]
