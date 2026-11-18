"""Tests for task 29 (drift report + spot checks) and 29b (rarest object first in the cap)."""
import json

from claudeknows import drift
from claudeknows import selflabel as sl
from claudeknows.detection_data import DetSample
from tests.test_selflabel import BOTTLE, SHOE, cut, make_walk, scripted

QUIET = {"log": lambda *_: None}


def _steady_walk(tmp_path, name="walk1", n=6):
    """Every frame: a bottle and a shoe both labellers agree on (all steady)."""
    d = make_walk(tmp_path, n=n, name=name)
    det = [[{"label": "water_bottle", "score": 0.9, "box": BOTTLE},
            {"label": "shoes", "score": 0.9, "box": SHOE}]] * n
    owl = [[("water_bottle", 0.8, BOTTLE), ("shoes", 0.8, SHOE)]] * n
    stats = sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, **QUIET)
    return d, stats


# ------------------------------------------------------------------ spot checks

def test_audit_picks_repeatable_and_only_agreed_boxes(tmp_path):
    d, stats = _steady_walk(tmp_path / "a")
    qs = [q for q in sl.load_questions(d) if q["why"] == sl.AUDIT]
    assert len(qs) == sl.AUDIT_PER_WALK == stats["audits"]
    labels = sl.load_labels(d)
    for q in qs:  # each audit question is one of that frame's accepted boxes
        assert {"label": q["label"], "box": q["box"]} in [
            {"label": b["label"], "box": b["box"]} for b in labels[q["frame"]]["boxes"]]
    # same walk name -> same picks
    d2, _ = _steady_walk(tmp_path / "b")
    picks = lambda dd: [(q["frame"], q["label"]) for q in sl.load_questions(dd) if q["why"] == sl.AUDIT]
    assert picks(d) == picks(d2)
    # audits never hold a frame back
    assert all(e["status"] == sl.TRAIN for e in labels.values())


def test_audit_no_removes_box_yes_keeps_it(tmp_path):
    d, _ = _steady_walk(tmp_path)
    qs = [q for q in sl.load_questions(d) if q["why"] == sl.AUDIT]
    no, yes = qs[0], qs[1]
    n_before = len(sl.load_labels(d)[no["frame"]]["boxes"])
    sl.apply_answers(d, {no["id"]: False, yes["id"]: True})
    labels = sl.load_labels(d)
    e = labels[no["frame"]]
    assert len(e["boxes"]) == n_before - 1 and e["status"] == sl.TRAIN
    assert no["label"] not in [b["label"] for b in e["boxes"]]
    assert yes["label"] in [b["label"] for b in labels[yes["frame"]]["boxes"]]
    rows = drift.write_audit_log(tmp_path, tmp_path / "models")
    assert sorted(r["yes"] for r in rows) == [False, True]
    assert json.loads((tmp_path / "models" / drift.AUDIT_LOG).read_text()) == rows


def test_audits_count_toward_cap(tmp_path):
    d = make_walk(tmp_path, n=6)
    det = [[{"label": "water_bottle", "score": 0.9, "box": BOTTLE}] for _ in range(6)]
    owl = [[("water_bottle", 0.8, BOTTLE), ("keys", 0.9, SHOE)] for _ in range(6)]
    stats = sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, max_questions=5, **QUIET)
    qs = sl.load_questions(d)
    assert len(qs) == 5 and sum(q["why"] == sl.AUDIT for q in qs) == 3
    assert stats["questions_over_cap"] == 4  # 6 key questions, only 2 places left


def test_wilson_range():
    p, lo, hi = drift.wilson(0, 20)
    assert p == 0 and lo == 0 and 0.15 < hi < 0.17
    assert drift.wilson(0, 0) is None


def test_audit_lines_warn_and_too_few():
    few = [{"object": "keys", "yes": False}] * 3
    lines, warns = drift.audit_lines(few)
    assert warns == [] and any("too few checks yet" in l for l in lines)
    many = [{"object": "keys", "yes": i % 3 != 0} for i in range(30)]  # 10 of 30 wrong
    lines, warns = drift.audit_lines(many)
    assert any(w.startswith("WARNING: keys") for w in warns)
    clean = [{"object": "shoes", "yes": True}] * 40
    assert drift.audit_lines(clean)[1] == []


# ------------------------------------------------------------------ agreement rate

def _series(rates, n=20):
    return [(f"w{i}", r, n) for i, r in enumerate(rates)]


def test_agreement_warning_rules():
    note, warn = drift.agreement_warning("keys", _series([0.6] * 5 + [0.3] * 3))
    assert warn and "keys" in warn and "agree less often" in warn
    assert drift.agreement_warning("keys", _series([0.6] * 6 + [0.3] * 2))[1] is None
    # walks with under 10 boxes are ignored
    small = _series([0.6] * 5) + [(f"s{i}", 0.1, 5) for i in range(3)]
    note, warn = drift.agreement_warning("keys", small)
    assert warn is None and "too few walks yet" in note


def test_agreement_counts_saved_per_walk(tmp_path):
    d = make_walk(tmp_path, n=3)
    det = [[{"label": "shoes", "score": 0.9, "box": SHOE}]] * 3
    owl = [[("shoes", 0.8, SHOE), ("keys", 0.9, BOTTLE)]] * 3
    sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, **QUIET)
    per = sl.load_stats(d)["per_object"]
    assert per["shoes"] == {"agreed": 3, "detector_only": 0, "owlv2_only": 0}
    assert per["keys"]["owlv2_only"] == 3
    series = drift.agreement_series(tmp_path)
    assert series["shoes"] == [("walk1", 1.0, 3)] and series["keys"] == [("walk1", 0.0, 3)]


# ------------------------------------------------------------------ the report

def test_report_and_history(tmp_path):
    _steady_walk(tmp_path / "walks")
    models = tmp_path / "models"
    drift.record_run(models, False)
    rows = drift.record_run(models, True, {"name": "c1", "promoted": True, "gold_created": "g1",
                                           "gold": {"current": 0.60, "candidate": 0.66}})
    rows = drift.record_run(models, True, {"name": "c2", "gold_created": "g1",
                                           "gold": {"current": 0.50, "candidate": 0.52}})
    assert len(rows) == 3
    lines = drift.report(tmp_path / "walks", models)
    text = "\n".join(lines)
    assert "current model scores 50.0%" in text  # lower than 60% on the same gold set
    assert "spot checks" in text and "too few walks yet" in text
    assert (models / drift.REPORT_FILE).exists() and (models / drift.AUDIT_LOG).exists()


def test_missing_object_warning():
    counts = [(f"w{i}", {"water_bottle": 5, "keys": 4 if i < 2 else 0, "shoes": 3}) for i in range(5)]
    warns = drift.missing_warnings(counts)
    assert len(warns) == 1 and "keys" in warns[0]
    assert drift.missing_warnings(counts[:3]) == []


# ------------------------------------------------------------------ 29b: rarest first

def _frames(label, n, start=0):
    return [DetSample(f"/x/{label}{start + i}.jpg", 0, "walk", "w", "together",
                      ((label, (0.1, 0.1, 0.5, 0.5)),), sl.SELF) for i in range(n)]


def test_cap_keeps_rare_keys_first():
    base = [object() for _ in range(70)]  # room for 30 self-labelled frames
    keys, shoes = _frames("keys", 10), _frames("shoes", 100)
    logged = []
    out = sl.cap_self_share(base, shoes + keys, 0.3, seed=1, log=logged.append)
    picked = out[70:]
    assert len(picked) == 30 and all(k in picked for k in keys)
    assert "self-labelled boxes used: water_bottle 0 of 0, keys 10 of 10, shoes 20 of 100" in logged
    # same seed -> same pick
    assert sl.cap_self_share(base, shoes + keys, 0.3, seed=1, log=lambda *_: None)[70:] == picked


def test_cap_under_limit_unchanged():
    base = [object() for _ in range(70)]
    extra = _frames("keys", 5) + _frames("shoes", 5)
    assert sl.cap_self_share(base, extra, 0.3, log=lambda *_: None) == base + extra


def test_cap_mixed_frame_joins_rarest_group():
    base = [object() for _ in range(7)]  # room for 3
    mixed = [DetSample("/x/m.jpg", 0, "walk", "w", "together",
                       (("shoes", (0, 0, 1, 1)), ("keys", (0, 0, 1, 1))), sl.SELF)]
    out = sl.cap_self_share(base, _frames("shoes", 20) + mixed, 0.3, log=lambda *_: None)
    assert mixed[0] in out[7:]
