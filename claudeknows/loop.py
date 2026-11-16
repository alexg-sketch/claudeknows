"""Phase 2, step 5: one learning-loop run (roadmap task 28).

Order of one run (the notebook `notebooks/learning_loop.ipynb` calls `run_loop`):
  1. import new walk videos into frames (task 24, `walks.import_walks`);
  2. auto-label the new walks (task 26, `selflabel.label_walk`);
  3. count the NEW training-approved boxes (agreed + steady, or "yes" answers from
     earlier walks). Too few -> no retraining this time ("not enough new data");
  4. otherwise retrain one candidate and run the gold-set gate (task 27);
  5. write a plain-language summary to Drive and list the review questions LAST.

Alex's yes/no answers are NOT waited for: in Colab "Run all" button clicks are only
handled after every cell has finished, so nothing in this run may depend on them.
Each click is saved at once (`answer`), and those frames count in the NEXT run
(RESEARCH.md 2026-11-12). That also keeps the loop able to run with no clicks at
all (phase 4).

Everything lives in Drive (`claudeknows_models/`), never in the public repo:
  walk_frames/            frames + selflabels.json + review.json per walk
  loop_state.json         which walk frames earlier retrains already used
  loop_summary.txt        the latest summary (plus loop_summary_YYYY-MM-DD.txt per run)
"""
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .objects import OBJECT_NAMES
from .selflabel import (HELD, LABELS_FILE, TRAIN, apply_answers, load_labels, load_questions,
                        walk_dirs)

MIN_NEW_BOXES = 30  # fewer new approved boxes than this in total -> no retrain
MIN_NEW_PER_OBJECT = 10  # ... or fewer than this for an object listed in MEANT_TO_COVER
MEANT_TO_COVER = ()  # objects every walk is meant to show, e.g. ("keys",); empty = total rule only
STATE_FILE = "loop_state.json"
SUMMARY_FILE = "loop_summary.txt"


def _read(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


# ------------------------------------------------------------------ new data

def trainable_boxes(walks_root):
    """{"walk/frame": [labels of its boxes]} for every frame that may be trained on now."""
    out = {}
    for d in walk_dirs(walks_root):
        for name, e in load_labels(d).items():
            if e["status"] == TRAIN:
                out[f"{d.name}/{name}"] = [b["label"] for b in e["boxes"]]
    return out


def used_frames(models_dir):
    return set(_read(Path(models_dir) / STATE_FILE, {"used": []})["used"])


def mark_used(models_dir, keys):
    """Remember the frames a retrain used, so the next run counts only newer ones."""
    used = used_frames(models_dir) | set(keys)
    _write(Path(models_dir) / STATE_FILE, {"used": sorted(used)})


def new_box_counts(walks_root, models_dir):
    """(Counter of new approved boxes per object, keys of the trainable frames)."""
    frames = trainable_boxes(walks_root)
    used = used_frames(models_dir)
    counts = Counter()
    for key, labels in frames.items():
        if key not in used:
            counts.update(labels)
    return counts, sorted(frames)


def should_retrain(counts, min_total=MIN_NEW_BOXES, min_per_object=MIN_NEW_PER_OBJECT,
                   covered=MEANT_TO_COVER):
    """(True/False, plain-language reason). Retraining costs a long Colab training and each
    retrain on a little self-labelled data can add errors, so skip it when the walks added
    too little (RESEARCH.md 2026-11-06). Skipped frames stay in the pool for next time."""
    total = sum(counts.values())
    per = ", ".join(f"{c} {counts.get(c, 0)}" for c in OBJECT_NAMES)
    if total < min_total:
        return False, (f"not enough new data ({total} boxes: {per}; at least {min_total} "
                       "needed); kept current model")
    short = [c for c in covered if counts.get(c, 0) < min_per_object]
    if short:
        return False, (f"not enough new data for {', '.join(short)} ({per}; at least "
                       f"{min_per_object} each needed); kept current model")
    return True, f"{total} new approved boxes ({per}): retraining"


# ------------------------------------------------------------------ review questions

def open_questions(walks_root):
    """Questions still waiting for a yes/no, oldest walk first. Each carries `walk_dir`.
    Only what the question view needs is used: photo, box, object (no score, no labeller)."""
    out = []
    for d in walk_dirs(walks_root):
        open_ids = {q for e in load_labels(d).values() for q in e["open"] if q}
        for q in load_questions(d):
            if q["id"] in open_ids:
                out.append({**q, "walk_dir": str(d)})
    return out


def answer(walks_root, qid, yes, log=print):
    """Save one yes/no answer straight away (called by the notebook's buttons)."""
    walk = qid.split(":")[0]
    d = Path(walks_root) / walk
    if not (d / LABELS_FILE).exists():
        raise ValueError(f"unknown walk for question {qid}")
    return apply_answers(d, {qid: bool(yes)}, log=log)


def question_view(walk_dir, q, max_side=480):
    """The photo with the one proposed box drawn on it (PIL image)."""
    from PIL import Image, ImageDraw
    with Image.open(Path(walk_dir) / q["frame"]) as im:
        im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    w, h = im.size
    x1, y1, x2, y2 = q["box"]
    ImageDraw.Draw(im).rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 40, 40),
                                 width=max(2, w // 150))
    return im


QUESTION = {"water_bottle": "Is this a water bottle?", "keys": "Are these keys?",
            "shoes": "Is this a shoe?"}


def question_text(q):
    return QUESTION.get(q["label"], f"Is this a {q['label'].replace('_', ' ')}?")


# ------------------------------------------------------------------ one run

def run_loop(walks_src, models_dir, *, label_fn=None, retrain_fn=None, frames_dir=None,
             import_fn=None, today=None, min_total=MIN_NEW_BOXES, covered=MEANT_TO_COVER,
             log=print):
    """One loop run. Returns {"summary": [lines], "retrained": bool, "result", "questions"}.

    label_fn(walk_dir) -> stats auto-labels one walk (Colab: current detector + OWLv2);
    None = no current model, so new walks wait. retrain_fn() -> `retrain.retrain` result
    dict (with "summary"); it reads the trainable walk frames itself.
    """
    from .walks import import_walks
    models_dir = Path(models_dir)
    frames_dir = Path(frames_dir) if frames_dir else models_dir / "walk_frames"
    today = today or date.today()
    lines = [f"Learning loop run {today.isoformat()}"]

    imported = (import_fn or import_walks)(walks_src, frames_dir)
    if imported:
        for r in imported:
            lines.append(f"new walk {r['name']}: {r['kept']} frames kept ({r['blurry']} blurry, "
                         f"{r['duplicates']} near-duplicates dropped)")
    else:
        lines.append("no new walk videos")

    waiting = [d for d in sorted(frames_dir.iterdir()) if d.is_dir()
               and not (d / LABELS_FILE).exists() and any(d.glob("*.jpg"))] if frames_dir.is_dir() else []
    for d in waiting:
        if label_fn is None:
            lines.append(f"{d.name}: not labelled yet (no current model; run the spatial notebook first)")
            continue
        s = label_fn(d)
        lines.append(f"{d.name}: {s['agreed']} boxes both labellers agreed on; frames: {s[TRAIN]} "
                     f"for training, {s[HELD]} waiting for your answers, {s['empty']} with nothing found")

    counts, keys = new_box_counts(frames_dir, models_dir)
    go, why = should_retrain(counts, min_total=min_total, covered=covered)
    lines.append(why)
    result = None
    if go:
        if retrain_fn is None:
            lines.append("retraining is switched off in this run")
            go = False
        else:
            result = retrain_fn()
            mark_used(models_dir, keys)
            lines += result.get("summary", [])

    qs = open_questions(frames_dir)
    lines.append(f"{len(qs)} question{'s' if len(qs) != 1 else ''} waiting for you (below); answers "
                 "are saved at once and will be used next time" if qs else "no questions waiting")
    text = "\n".join(lines) + "\n"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / SUMMARY_FILE).write_text(text)
    (models_dir / f"loop_summary_{today.isoformat()}.txt").write_text(text)
    log(text)
    return {"summary": lines, "retrained": go, "result": result, "questions": qs}


def colab_label_fn(checkpoint):
    """label_fn for Colab: the current detector + OWLv2 (needs `transformers`)."""
    from .autobox import owl_detector
    from .detector import _cut, detect, load_detector
    from .selflabel import LOW_SCORE, label_walk, review_loss
    model, classes = load_detector(checkpoint)
    owl = owl_detector()
    return lambda d: label_walk(d, lambda im: detect(model, im, classes, min_score=LOW_SCORE), owl,
                                lambda l: _cut(model, l, None),
                                loss_fn=lambda im, boxes: review_loss(model, im, boxes, classes))


def prepare_training_photos(root, photos_dir=None, cap=300, run=None):
    """Rebuild the original training photos on Colab's (temporary) disk, the same steps as the
    spatial notebook: Open Images download, Alex's Drive footage, OWLv2 boxes. Only called
    when a retrain will really happen (it takes a while)."""
    import subprocess
    import sys
    run = run or (lambda cmd: subprocess.run(cmd, check=True))
    py = [sys.executable, "-m"]
    run(py + ["claudeknows.detection_data", "download", "--out", str(root), "--cap", str(cap)])
    if photos_dir and Path(photos_dir).is_dir():
        run(py + ["claudeknows.detection_data", "import", str(photos_dir), "--root", str(root)])
    run(py + ["claudeknows.autobox", "run", str(root), "--threshold", "0.3"])
    run(py + ["claudeknows.autobox", "fill", str(root), "--threshold", "0.4"])


def colab_retrain_fn(root, models_dir, frames_dir, photos_dir=None, gold_dir=None, epochs=15,
                     workers=2, noise_floor=True, log=print):
    """retrain_fn for Colab: rebuild the original photos in `root`, measure any missing
    noise-floor seeds (once ever; only with a gold set), then train + gate one candidate on
    the original photos + approved walk frames. The gold set lives in `gold_dir` (Drive)."""
    from argparse import Namespace
    from . import retrain
    from .selflabel import self_samples

    def run():
        prepare_training_photos(root, photos_dir)
        a = Namespace(root=root, gold_dir=gold_dir, epochs=epochs, batch_size=4, patience=4,
                      workers=workers, max_batches=None)
        base, gsamps, gold, kwargs = retrain._data(a, log)
        if noise_floor and gsamps:
            missing = retrain.noise_progress(models_dir, retrain.recipe_of(kwargs), log=log)[2]
            if missing:
                log(f"measuring the noise floor first: {len(missing)} short training(s), saved one by one")
                retrain.measure_noise_floor(base, gsamps, models_dir, gold_manifest=gold,
                                            train_kwargs=kwargs, log=log)
        extra = self_samples(frames_dir, gold, log)
        return retrain.retrain(base, extra, gsamps, models_dir, gold_manifest=gold,
                               train_kwargs=kwargs, log=log)
    return run
