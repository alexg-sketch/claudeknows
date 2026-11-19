"""The gold-set notebook's helpers: which videos are offered, answers, freezing."""
import json
from pathlib import Path

from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import gold

NB = Path(__file__).resolve().parents[1] / "notebooks" / "gold_set.ipynb"
QUIET = dict(log=lambda *_: None)


def _videos(root, spec, frames=10):
    """spec: {folder: [video names]}; each video gets its own colour (= its own scene)."""
    data_by_folder, colour = {}, 0
    for folder, vids in spec.items():
        d = root / "own" / folder
        d.mkdir(parents=True, exist_ok=True)
        data = {}
        for v in vids:
            colour += 40
            for i in range(frames):
                name = f"vid_{v}__f{i:04d}.jpg"
                Image.new("RGB", (64, 48), (colour % 256, (colour * 3) % 256, 90)).save(d / name)
                label = "keys" if folder == "together" else folder
                data[name] = {"by": "owlv2", "boxes": [{"label": label, "box": [0.2, 0.2, 0.6, 0.7]}]}
        dd.save_boxes(d, data)
        data_by_folder[folder] = data
    return dd.scan_detection(root)[0]


def test_never_takes_the_only_video_of_a_folder_and_prefers_together(tmp_path):
    root = tmp_path / "det"
    samples = _videos(root, {"keys": ["k1", "k2"], "shoes": ["s1"], "together": ["t1", "t2", "t3"]})
    logs = []
    cands = gold.pick_gold_scenes(samples, per_scene=4, log=logs.append)
    groups = {s.group for s in cands}
    assert "vid_s1" not in groups  # the only shoes video stays for training
    assert sum(g.startswith("vid_t") for g in groups) == 2  # together first, one kept back
    assert sum(g.startswith("vid_k") for g in groups) == 1
    assert all(len([s for s in cands if s.group == g]) == 4 for g in groups)  # spread frames
    assert any("shoes: only one scene" in l for l in logs)
    assert any("needs 5 gold scenes" in l for l in logs)  # only 3 places to offer
    assert gold.pick_gold_scenes(samples, per_scene=4, **QUIET) == cands  # same pick every time


def test_look_alike_videos_are_offered_together(tmp_path):
    root = tmp_path / "det"
    samples = _videos(root, {"keys": ["k1", "k2", "k3"]})
    scenes = {s.group: s.group for s in samples}
    scenes["vid_k1"] = scenes["vid_k2"] = "scene_vid_k1"  # same room
    cands = gold.pick_gold_scenes(samples, scenes, per_scene=30, **QUIET)
    groups = {s.group for s in cands}
    assert groups in ({"vid_k1", "vid_k2"}, {"vid_k3"})  # a scene is never split


def test_answers_freeze_only_yes_and_refuse_small_or_existing(tmp_path):
    root, drive = tmp_path / "det", tmp_path / "drive_gold"
    samples = _videos(root, {"keys": [f"k{i}" for i in range(7)]}, frames=2)
    cands = gold.pick_gold_scenes(samples, target=6, per_scene=2, **QUIET)
    path = tmp_path / "answers.json"
    assert gold.freeze_from_answers(cands, {}, root, drive, **QUIET) is None  # nothing answered
    for s in cands[:4]:  # 2 scenes only
        gold.save_answer(path, s.path.name, True)
    gold.save_answer(path, cands[4].path.name, False)
    answers = gold.load_answers(path)
    assert answers[cands[4].path.name] is False and len(answers) == 5
    assert gold.freeze_from_answers(cands, answers, root, drive, **QUIET) is None  # < 5 scenes
    for s in cands[5:]:
        gold.save_answer(path, s.path.name, True)
    m = gold.freeze_from_answers(cands, gold.load_answers(path), root, drive, **QUIET)
    assert len(m["images"]) == len(cands) - 1 and all(e["by"] == "checked" for e in m["images"])
    assert cands[4].path.name not in {Path(e["path"]).name for e in m["images"]}
    assert gold.load_gold(drive)["images"] == m["images"]
    assert gold.freeze_from_answers(cands, answers, root, drive, **QUIET) is None  # frozen for good


def test_training_finds_the_drive_gold_set_by_itself(tmp_path):
    root, drive = tmp_path / "det", tmp_path / "drive_gold"
    root.mkdir()
    assert gold.default_gold_dir(root, drive) == root  # no gold anywhere
    drive.mkdir()
    (drive / gold.GOLD_FILE).write_text("{}")
    assert gold.default_gold_dir(root, drive) == drive
    (root / gold.GOLD_FILE).write_text("{}")
    assert gold.default_gold_dir(root, drive) == root  # a local one wins


def test_gold_notebook_uses_real_helpers_and_no_personal_data():
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for name in ("claudeknows.detection_data import", "claudeknows.autobox run", "pick_gold_scenes",
                 "save_answer", "freeze_from_answers", "claudeknows_models/gold", "drive.mount"):
        assert name in code
    for name in ("pick_gold_scenes", "save_answer", "load_answers", "freeze_from_answers", "gold_view"):
        assert hasattr(gold, name)
    lines = [l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
             for l in code.splitlines()]
    compile("\n".join(lines), "notebook", "exec")
    assert "git add" not in code and "git push" not in code
