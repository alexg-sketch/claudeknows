import json
from collections import Counter
from datetime import date
from pathlib import Path

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
    loop.prepare_training_photos("data/detection", tmp_path / "missing", run=cmds.append, log=QUIET)
    steps = [" ".join(c[2:4]) for c in cmds]
    assert steps == ["claudeknows.detection_data download", "claudeknows.autobox run",
                     "claudeknows.autobox fill"]  # no Drive folder -> no import step
    cmds.clear()
    loop.prepare_training_photos("data/detection", tmp_path, run=cmds.append, log=QUIET)
    assert [" ".join(c[2:4]) for c in cmds][1] == "claudeknows.detection_data import"


# ------------------------------------------------------------------ task 29d: snapshot

class FakeBuild:
    """Fake `run`: records commands; the download writes a few tiny photos into --out."""

    def __init__(self):
        self.cmds = []

    def __call__(self, cmd):
        self.cmds.append(cmd)
        if cmd[3] == "download":
            root = Path(cmd[cmd.index("--out") + 1])
            for obj in ("keys", "shoes"):
                (root / "web" / obj).mkdir(parents=True, exist_ok=True)
                (root / "web" / obj / f"{obj}1.jpg").write_bytes(obj.encode() * 50)
                (root / "web" / obj / "boxes.json").write_text('{"x": 1}')

    def steps(self):
        return [" ".join(c[2:4]) for c in self.cmds]


def _footage(tmp_path):
    photos = tmp_path / "claudeknows_photos"
    (photos / "keys").mkdir(parents=True)
    (photos / "keys" / "a.MOV").write_bytes(b"video")
    (photos / "walks").mkdir()
    (photos / "walks" / "walk1.MOV").write_bytes(b"walk")  # walks never count
    return photos


def _snap_tree(root):
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in Path(root).rglob("*") if f.is_file()}


def test_snapshot_built_once_then_reused_with_identical_files(tmp_path):
    photos, snap, run = _footage(tmp_path), tmp_path / "drive" / "training_set", FakeBuild()
    first = loop.prepare_training_photos(tmp_path / "a", photos, snap, run=run, log=QUIET)
    assert not first["reused"] and "no saved snapshot yet" in first["message"]
    assert (snap / loop.SNAPSHOT_TAR).exists() and (snap / loop.SNAPSHOT_MANIFEST).exists()
    assert not list(snap.glob("*.tmp"))
    man = json.loads((snap / loop.SNAPSHOT_MANIFEST).read_text())
    assert man["photos"] == 2 and man["counts"] == {"web/keys": 1, "web/shoes": 1}
    assert man["footage"] == [["keys/a.MOV", 5]] and man["settings"]["cap"] == 300
    assert first["snapshot"] == loop.snapshot_id(snap) and len(first["snapshot"]) == 12

    run2 = FakeBuild()
    second = loop.prepare_training_photos(tmp_path / "b", photos, snap, run=run2, log=QUIET)
    assert run2.cmds == []  # no download, import or OWLv2
    assert second["reused"] and second["snapshot"] == first["snapshot"]
    assert second["message"] == f"training photos: snapshot {first['snapshot']} (2 photos) reused"
    assert _snap_tree(tmp_path / "b") == _snap_tree(tmp_path / "a")


def test_damaged_snapshot_rebuilds_with_message(tmp_path):
    import tarfile
    photos, snap = _footage(tmp_path), tmp_path / "training_set"
    loop.prepare_training_photos(tmp_path / "a", photos, snap, run=FakeBuild(), log=QUIET)
    # replace one photo inside the tar with different bytes (manifest hash no longer matches)
    src = tmp_path / "evil"
    src.mkdir()
    (src / "keys1.jpg").write_bytes(b"different")
    with tarfile.open(snap / loop.SNAPSHOT_TAR, "a") as tar:
        tar.add(src / "keys1.jpg", arcname="web/keys/keys1.jpg")
    run = FakeBuild()
    out = loop.prepare_training_photos(tmp_path / "b", photos, snap, run=run, log=QUIET)
    assert not out["reused"] and "damaged" in out["message"] and "web/keys/keys1.jpg" in out["message"]
    assert run.steps()[0] == "claudeknows.detection_data download"
    # the rebuilt snapshot is reused next time
    assert loop.prepare_training_photos(tmp_path / "c", photos, snap, run=FakeBuild(), log=QUIET)["reused"]


def test_new_own_footage_file_rebuilds(tmp_path):
    photos, snap = _footage(tmp_path), tmp_path / "training_set"
    first = loop.prepare_training_photos(tmp_path / "a", photos, snap, run=FakeBuild(), log=QUIET)
    (photos / "walks" / "walk2.MOV").write_bytes(b"walk")  # a new walk alone: still reused
    assert loop.prepare_training_photos(tmp_path / "b", photos, snap, run=FakeBuild(), log=QUIET)["reused"]
    (photos / "keys" / "b.MOV").write_bytes(b"new video")
    run = FakeBuild()
    out = loop.prepare_training_photos(tmp_path / "c", photos, snap, run=run, log=QUIET)
    assert not out["reused"] and "1 new, e.g. keys/b.MOV" in out["message"]
    assert "claudeknows.detection_data import" in run.steps()
    assert out["snapshot"] != first["snapshot"]


def test_changed_photo_settings_rebuild_and_name_the_setting(tmp_path, monkeypatch):
    photos, snap = _footage(tmp_path), tmp_path / "training_set"
    loop.prepare_training_photos(tmp_path / "a", photos, snap, run=FakeBuild(), log=QUIET)
    assert loop.prepare_training_photos(tmp_path / "b", photos, snap, run=FakeBuild(), log=QUIET)["reused"]
    out = loop.prepare_training_photos(tmp_path / "c", photos, snap, fps=3.0, run=FakeBuild(), log=QUIET)
    assert not out["reused"] and "photo settings changed (fps)" in out["message"]
    monkeypatch.setattr(loop, "PHOTO_PIPELINE_VERSION", loop.PHOTO_PIPELINE_VERSION + 1)
    out = loop.prepare_training_photos(tmp_path / "d", photos, snap, fps=3.0, run=FakeBuild(), log=QUIET)
    assert not out["reused"] and "(PHOTO_PIPELINE_VERSION)" in out["message"]


def test_old_manifest_without_new_keys_rebuilds_once(tmp_path):
    photos, snap = _footage(tmp_path), tmp_path / "training_set"
    loop.prepare_training_photos(tmp_path / "a", photos, snap, run=FakeBuild(), log=QUIET)
    man = json.loads((snap / loop.SNAPSHOT_MANIFEST).read_text())
    for k in ("fps", "DETECT_SIDE", "BLUR_THRESHOLD", "owl_model", "PHOTO_PIPELINE_VERSION"):
        del man["settings"][k]
    (snap / loop.SNAPSHOT_MANIFEST).write_text(json.dumps(man))
    out = loop.prepare_training_photos(tmp_path / "b", photos, snap, run=FakeBuild(), log=QUIET)
    assert not out["reused"]
    assert "(BLUR_THRESHOLD, DETECT_SIDE, PHOTO_PIPELINE_VERSION, fps, owl_model)" in out["message"]
    assert loop.prepare_training_photos(tmp_path / "c", photos, snap, run=FakeBuild(), log=QUIET)["reused"]


def test_snapshot_id_changes_recipe_and_is_saved_with_version(tmp_path):
    from claudeknows import retrain
    kw = {"epochs": 15}
    assert retrain.recipe_of(kw, training_set="aaa") != retrain.recipe_of(kw, training_set="bbb")
    assert "training_set" not in retrain.recipe_of(kw)  # no snapshot: recipe as before
    from claudeknows.detector import build_detector
    m = build_detector(pretrained=False)
    name = retrain.save_version(m, tmp_path, extra={"init": "coco", "training_set": "abc123"})
    entry = retrain.load_versions(tmp_path)["all"][-1]
    assert entry["name"] == name and entry["training_set"] == "abc123"


def test_colab_retrain_fn_passes_snapshot_id_to_noise_floor_and_candidate(tmp_path, monkeypatch):
    from claudeknows import retrain
    seen = {}
    monkeypatch.setattr(loop, "prepare_training_photos",
                        lambda root, photos, snap, log: {"snapshot": "abc123def456", "reused": True,
                                                         "photos": 2, "message": "training photos: reused"})
    monkeypatch.setattr(retrain, "_data", lambda a, log: ([], ["g"], {"created": "x"}, {"epochs": 1}))
    monkeypatch.setattr(retrain, "measure_noise_floor",
                        lambda *a, training_set=None, **k: seen.setdefault("floor", training_set))
    monkeypatch.setattr(retrain, "retrain", lambda *a, training_set=None, **k: (
        seen.setdefault("retrain", training_set), {"summary": ["candidate x"], "decision": None})[1])
    monkeypatch.setattr("claudeknows.selflabel.self_samples", lambda *a: [])
    out = loop.colab_retrain_fn(tmp_path / "root", tmp_path / "models", tmp_path / "frames", log=QUIET)()
    assert seen == {"floor": "abc123def456", "retrain": "abc123def456"}
    assert out["summary"] == ["training photos: reused", "candidate x"]
    assert out["training_set"] == "abc123def456"
