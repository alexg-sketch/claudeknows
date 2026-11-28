"""Tests for task 29f: labeller check on the gold set (report only)."""
import copy
import json
from datetime import date

from PIL import Image

from claudeknows import autobox, drift, gold, loop
from claudeknows.detection_data import DetSample, save_boxes

QUIET = lambda *_: None  # noqa: E731
BOTTLE = (0.1, 0.1, 0.4, 0.6)
KEYS = (0.6, 0.6, 0.8, 0.8)
FAR = (0.0, 0.8, 0.1, 0.9)  # overlaps nothing
CUT = {"water_bottle": 0.5, "keys": 0.5, "shoes": 0.5}
det_cut = CUT.get


def _gold(tmp_path, n=10, boxes=(("water_bottle", BOTTLE), ("keys", KEYS))):
    """n gold photos (one per video, folder "together"), each with `boxes`, plus a manifest."""
    root = tmp_path / "gold"
    (root / "own" / "together").mkdir(parents=True, exist_ok=True)
    samps = []
    for i in range(n):
        p = root / "own" / "together" / f"v{i}_f0.jpg"
        Image.new("RGB", (64, 48), (10 * i, 50, 90)).save(p)
        samps.append(DetSample(p, "together", "own", f"v{i}", "together", tuple(boxes), gold.CHECKED))
    manifest = gold.build_gold(samps, root, distance=False, log=QUIET)
    return manifest, gold.gold_samples(manifest, root)


def det_of(found):
    return lambda im: [{"label": l, "score": s, "box": b} for l, s, b in found]


def owl_of(found):
    return lambda im, labels: list(found)


GOLD_FOUND = [("water_bottle", 0.9, BOTTLE), ("keys", 0.9, KEYS)]


def test_perfect_labellers_score_one_of_one(tmp_path):
    _, samps = _gold(tmp_path, n=1)
    r = drift.labeller_check(samps, det_of(GOLD_FOUND), owl_of(GOLD_FOUND), det_cut)
    assert r["photos"] == 1
    for key in ("detector", "agreement"):
        assert r[key]["water_bottle"] == [1, 1, 1] and r[key]["keys"] == [1, 1, 1]
        assert r[key]["shoes"] == [0, 0, 0]


def test_extra_wrong_box_halves_precision_keeps_recall(tmp_path):
    _, samps = _gold(tmp_path, n=10)
    det = det_of(GOLD_FOUND + [("water_bottle", 0.8, FAR), ("keys", 0.7, FAR)])
    r = drift.labeller_check(samps, det, owl_of(GOLD_FOUND), det_cut)
    assert r["detector"]["water_bottle"] == [10, 20, 10]
    lines = drift.labeller_lines(r)
    line = next(l for l in lines if l.startswith("  water_bottle, current detector"))
    assert "precision 10 of 20 (50%" in line and "recall 10 of 10 (100%" in line
    assert "range" in line


def test_wrong_label_counts_as_wrong(tmp_path):
    _, samps = _gold(tmp_path, n=1)
    r = drift.labeller_check(samps, det_of([("shoes", 0.9, BOTTLE)]), owl_of([]), det_cut)
    assert r["detector"]["shoes"] == [0, 1, 0] and r["detector"]["water_bottle"] == [0, 0, 1]


def test_two_proposals_on_one_gold_box_count_once(tmp_path):
    _, samps = _gold(tmp_path, n=1, boxes=(("water_bottle", BOTTLE),))
    near = (0.11, 0.1, 0.41, 0.6)
    r = drift.labeller_check(samps, det_of([("water_bottle", 0.9, BOTTLE), ("water_bottle", 0.8, near)]),
                             owl_of([]), det_cut)
    assert r["detector"]["water_bottle"] == [1, 2, 1]


def test_agreement_needs_both_and_below_cutoff_is_ignored(tmp_path):
    _, samps = _gold(tmp_path, n=1)
    # OWLv2 sees only the bottle; the detector's keys box is below its cut-off
    r = drift.labeller_check(samps, det_of([("water_bottle", 0.9, BOTTLE), ("keys", 0.4, KEYS)]),
                             owl_of([("water_bottle", 0.9, BOTTLE), ("keys", 0.9, KEYS)]), det_cut)
    assert r["agreement"]["water_bottle"] == [1, 1, 1]
    assert r["agreement"]["keys"] == [0, 0, 1] and r["detector"]["keys"] == [0, 0, 1]


def test_owl_duplicates_removed_before_agreement(tmp_path):
    _, samps = _gold(tmp_path, n=1, boxes=(("water_bottle", BOTTLE),))
    dup = (0.1, 0.1, 0.41, 0.61)
    r = drift.labeller_check(samps, det_of([("water_bottle", 0.9, BOTTLE), ("water_bottle", 0.9, dup)]),
                             owl_of([("water_bottle", 0.9, BOTTLE), ("water_bottle", 0.85, dup)]), det_cut)
    # without NMS both detector boxes would pair with an OWLv2 box -> 2 proposals
    assert r["agreement"]["water_bottle"] == [1, 1, 1]


def test_too_few_gold_boxes_and_caveat(tmp_path):
    _, samps = _gold(tmp_path, n=3)
    lines = drift.labeller_lines(drift.labeller_check(samps, det_of(GOLD_FOUND), owl_of(GOLD_FOUND), det_cut))
    assert any("too few gold boxes for keys (3;" in l for l in lines)
    assert any("too few gold boxes for shoes (0;" in l for l in lines)
    assert any("measured on photos OWLv2 already got fully right" in l for l in lines)


def test_owlv2_never_scored_against_gold(tmp_path):
    manifest, samps = _gold(tmp_path, n=10)
    models = tmp_path / "models"
    lines = drift.run_labeller_check(models, manifest, samps, "det1",
                                     lambda: (det_of(GOLD_FOUND), owl_of(GOLD_FOUND), det_cut), log=QUIET)
    assert not any("owlv2" in l.lower() and "precision" in l for l in lines)
    assert any(l.startswith("OWLv2 on your footage: no gold-set answers") for l in lines)


def test_owl_footage_from_answers_old_and_new_format(tmp_path):
    manifest, _ = _gold(tmp_path, n=1)
    own = tmp_path / "own"
    (own / "keys").mkdir(parents=True)
    entries = {f"k{i}.jpg": {"by": "owlv2", "boxes": []} for i in range(10)}
    entries.update({f"r{i}.jpg": {"by": autobox.REJECTED, "boxes": []} for i in range(7)})
    save_boxes(own / "keys", entries)
    answers = {f"k{i}.jpg": i < 6 for i in range(10)}  # old format: 6 Yes, 4 No
    answers["v0_f0.jpg"] = True  # folder from the gold manifest
    answers["lost.jpg"] = False  # no folder info anywhere
    answers["s1.jpg"] = {"ok": False, "why": "missing", "folder": "shoes"}  # newer format
    answers["_rejected"] = {"shoes": 4}
    rows = drift.owl_footage(answers, manifest, own)
    assert rows["keys"] == {"yes": 6, "no": 4, "wrong": 0, "missing": 0, "rejected": 7}
    assert rows["together"]["yes"] == 1 and rows["unknown"]["no"] == 1
    assert rows["shoes"] == {"yes": 0, "no": 1, "wrong": 0, "missing": 1, "rejected": 4}
    assert "_rejected" not in rows
    lines, sugg = drift.owl_footage_lines(rows)
    keys = next(l for l in lines if l.startswith("  keys:"))
    assert "6 of 10 (60%, range" in keys and "7 footage frames where OWLv2 found nothing" in keys
    assert any("keys" in s and "Nothing was changed" in s for s in sugg)
    # no own folder on disk -> works, rejected unknown
    rows = drift.owl_footage({"k1.jpg": True}, None, tmp_path / "nowhere")
    assert rows == {"unknown": {"yes": 1, "no": 0, "wrong": 0, "missing": 0, "rejected": None}}


def test_missing_answers_suggest_lower_cutoff_only_as_text():
    rows = {"keys": {"yes": 2, "no": 6, "wrong": 1, "missing": 5, "rejected": None}}
    _, sugg = drift.owl_footage_lines(rows)
    assert len(sugg) == 1 and "lower OWLv2 cut-off for keys" in sugg[0] and "nothing was changed" in sugg[0]


def test_cache_skips_labellers_until_gold_or_detector_changes(tmp_path):
    manifest, samps = _gold(tmp_path, n=2)
    models = tmp_path / "models"
    calls = []

    def make():
        calls.append(1)
        return det_of(GOLD_FOUND), owl_of(GOLD_FOUND), det_cut

    first = drift.run_labeller_check(models, manifest, samps, "det1", make, log=QUIET)
    again = drift.run_labeller_check(models, manifest, samps, "det1", make, log=QUIET)
    assert len(calls) == 1 and first == again
    drift.run_labeller_check(models, manifest, samps, "det2", make, log=QUIET)
    assert len(calls) == 2
    other = copy.deepcopy(manifest)
    other["images"][0]["sha256"] = "0" * 64
    drift.run_labeller_check(models, other, samps, "det2", make, log=QUIET)
    assert len(calls) == 3
    # new answers show up without re-running the labellers
    (models / drift.GOLD_ANSWERS).write_text(json.dumps({"v0_f0.jpg": True}))
    lines = drift.run_labeller_check(models, other, samps, "det2", make, log=QUIET)
    assert len(calls) == 3 and any(l.startswith("  together: 1 of 1") for l in lines)


def test_thresholds_and_rules_unchanged(tmp_path):
    manifest, samps = _gold(tmp_path, n=10)
    before = (copy.deepcopy(autobox.THRESHOLD), dict(CUT))
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / drift.GOLD_ANSWERS).write_text(json.dumps({f"k{i}": {"ok": False, "why": "missing",
                                                                          "folder": "keys"} for i in range(9)}))
    drift.run_labeller_check(tmp_path / "m", manifest, samps, "d",
                             lambda: (det_of([]), owl_of(GOLD_FOUND), det_cut), log=QUIET)
    assert (autobox.THRESHOLD, CUT) == before


def test_loop_summary_and_drift_report_show_labeller_check(tmp_path):
    manifest, samps = _gold(tmp_path, n=10)
    models = tmp_path / "models"
    fn = lambda: drift.run_labeller_check(models, manifest, samps, "d",  # noqa: E731
                                          lambda: (det_of(GOLD_FOUND), owl_of(GOLD_FOUND), det_cut), log=QUIET)
    kw = dict(import_fn=lambda *_: [], log=QUIET, today=date(2026, 11, 27))
    out = loop.run_loop(tmp_path / "walks", models, labeller_fn=fn, **kw)
    assert any(l.startswith("labeller check on 10 gold photos") for l in out["summary"])
    assert "labeller check on 10 gold photos" in (models / drift.REPORT_FILE).read_text()

    def broken():
        raise RuntimeError("no GPU")
    out = loop.run_loop(tmp_path / "walks", models, labeller_fn=broken, **kw)
    assert "labeller check skipped: no GPU" in out["summary"]


def test_colab_labeller_fn_without_gold_set(tmp_path):
    lines = loop.colab_labeller_fn(tmp_path / "m", tmp_path / "nogold", tmp_path / "det.pt", log=QUIET)()
    assert lines == []
