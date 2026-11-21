from collections import Counter
from datetime import date

from claudeknows import loop
from claudeknows import selflabel as sl
from tests.test_selflabel import BOTTLE, cut, make_walk, scripted

QUIET = lambda *_: None  # noqa: E731


def _moving(i):
    x, y = (i % 9) * 0.1, 0.55 + (i // 9 % 4) * 0.1
    return (x, y, x + 0.08, y + 0.08)


def _label_fn(n, shoe_frames, audits=0):
    """Fake labellers: a steady bottle both see in every frame; a shoe only OWLv2 sees
    in `shoe_frames` (so those frames become questions and are held back). The shoe moves
    between frames, so each frame is its own question (no grouping into tracks, task 29c)."""
    det = [[{"label": "water_bottle", "score": 0.9, "box": BOTTLE}] for _ in range(n)]
    owl = [[("water_bottle", 0.8, BOTTLE)] + ([("shoes", 0.7, _moving(i))] if i in shoe_frames else [])
           for i in range(n)]
    return lambda d: sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, audit_per_walk=audits,
                                   log=QUIET)


def test_should_retrain_29_skipped_30_retrains():
    go, why = loop.should_retrain(Counter(water_bottle=20, shoes=9))
    assert not go and why.startswith("not enough new data (29 boxes") and "kept current model" in why
    go, why = loop.should_retrain(Counter(water_bottle=20, shoes=10))
    assert go and why.startswith("30 new approved boxes")
    # an object the walks are meant to cover needs 10 new boxes of its own
    go, why = loop.should_retrain(Counter(water_bottle=40, keys=9), covered=("keys",))
    assert not go and "for keys" in why
    assert loop.should_retrain(Counter(water_bottle=40, keys=10), covered=("keys",))[0]


def test_loop_finishes_without_answers_and_answers_count_next_run(tmp_path):
    models, frames, n = tmp_path / "models", tmp_path / "models" / "walk_frames", 40
    make_walk(frames, n=n, name="walk1")
    calls = []

    def retrain_fn():
        calls.append(len(sl.self_samples(frames, log=QUIET)))
        return {"summary": ["kept the current model: (fake gate)"]}

    kw = dict(retrain_fn=retrain_fn, import_fn=lambda *_: [], log=QUIET, today=date(2026, 11, 16))
    # run 1: 25 frames trainable (25 bottle boxes) -> too little, no retrain; 15 questions
    out = loop.run_loop(tmp_path / "walks", models, label_fn=_label_fn(n, range(15)), **kw)
    assert not out["retrained"] and calls == []
    assert any(l.startswith("not enough new data (25 boxes") for l in out["summary"])
    assert len(out["questions"]) == 15 and "15 questions waiting" in out["summary"][-1]
    text = (models / loop.SUMMARY_FILE).read_text()
    assert text == (models / "loop_summary_2026-11-16.txt").read_text()
    assert "will be used next time" in text

    # Alex clicks "yes" on every question after the run: saved at once
    for q in out["questions"]:
        loop.answer(frames, q["id"], True, log=QUIET)
    assert loop.open_questions(frames) == []

    # run 2: the walk is not labelled again; 40 bottle + 15 shoe boxes are new -> retrain
    out = loop.run_loop(tmp_path / "walks", models, label_fn=None, **kw)
    assert out["retrained"] and calls == [40]
    assert "kept the current model: (fake gate)" in out["summary"]
    assert out["summary"][-1] == "no questions waiting"

    # run 3: nothing new since the last retrain -> skipped
    out = loop.run_loop(tmp_path / "walks", models, label_fn=None, **kw)
    assert not out["retrained"] and calls == [40]
    assert any(l.startswith("not enough new data (0 boxes") for l in out["summary"])


def test_loop_spot_checks_and_drift_report(tmp_path):
    models, frames = tmp_path / "models", tmp_path / "models" / "walk_frames"
    make_walk(frames, n=6, name="walk1")
    kw = dict(import_fn=lambda *_: [], log=QUIET, today=date(2026, 11, 18))
    out = loop.run_loop(tmp_path / "walks", models, label_fn=_label_fn(6, {2}, audits=3), **kw)
    whys = sorted(q["why"] for q in out["questions"])
    assert whys.count(sl.AUDIT) == 3 and len(whys) == 4
    assert any(l.startswith("Drift report") for l in out["summary"])
    assert "3 of them spot checks" not in out["summary"]  # the label line stays short
    # a "no" on a spot check removes the box and is logged
    audit = next(q for q in out["questions"] if q["why"] == sl.AUDIT)
    loop.answer(frames, audit["id"], False, log=QUIET)
    out = loop.run_loop(tmp_path / "walks", models, label_fn=None, **kw)
    assert any("1 of 1 spot-checked auto-labels wrong" in l for l in out["summary"])
    assert len(out["questions"]) == 3
    assert (models / "drift_history.json").exists()


def test_new_walk_waits_without_a_current_model(tmp_path):
    models = tmp_path / "models"
    make_walk(models / "walk_frames", n=3, name="walk1")
    out = loop.run_loop(tmp_path / "walks", models, import_fn=lambda *_: [], log=QUIET)
    assert any("not labelled yet" in l for l in out["summary"])
    assert not out["retrained"]


def test_question_view_shows_only_photo_box_and_object(tmp_path):
    d = make_walk(tmp_path, n=5)
    _label_fn(5, {2})(d)
    (q,) = loop.open_questions(tmp_path)
    assert loop.question_text(q) == "Is this a shoe?"
    im = loop.question_view(q["walk_dir"], q)
    assert im.size[0] <= 480
    text = loop.question_text(q)
    assert str(q["score"]) not in text and "owl" not in text.lower()
    counts = loop.answer(tmp_path, q["id"], False, log=QUIET)
    assert counts[sl.HELD] == 0


def test_prepare_training_photos_same_steps_as_spatial_notebook(tmp_path):
    cmds = []
    loop.prepare_training_photos("data/detection", tmp_path / "missing", run=cmds.append)
    steps = [" ".join(c[2:4]) for c in cmds]
    assert steps == ["claudeknows.detection_data download", "claudeknows.autobox run",
                     "claudeknows.autobox fill"]  # no Drive folder -> no import step
    cmds.clear()
    loop.prepare_training_photos("data/detection", tmp_path, run=cmds.append)
    assert [" ".join(c[2:4]) for c in cmds][1] == "claudeknows.detection_data import"
