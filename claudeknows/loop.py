"""Phase 2, step 5: one learning-loop run (roadmap task 28).

Order of one run (the notebook `notebooks/learning_loop.ipynb` calls `run_loop`):
  1. import new walk videos into frames (task 24, `walks.import_walks`);
  2. auto-label the new walks (task 26, `selflabel.label_walk`);
  3. count the NEW training-approved boxes (agreed + steady, or "yes" answers from
     earlier walks). Too few -> no retraining this time ("not enough new data");
  4. otherwise retrain one candidate and run the gold-set gate (task 27);
  5. add the drift report (task 29, `drift.py`), write a plain-language summary to
     Drive and list the review questions LAST.

Alex's yes/no answers are NOT waited for: in Colab "Run all" button clicks are only
handled after every cell has finished, so nothing in this run may depend on them.
Each click is saved at once (`answer`), and those frames count in the NEXT run
(RESEARCH.md 2026-11-12). That also keeps the loop able to run with no clicks at
all (phase 4).

Everything lives in Drive (`claudeknows_models/`), never in the public repo:
  walk_frames/            frames + selflabels.json + review.json per walk
  loop_state.json         which walk frames earlier retrains already used
  loop_summary.txt        the latest summary (plus loop_summary_YYYY-MM-DD.txt per run)
  drift_history.json      one row per run (gold scores), for the drift report (task 29)
  drift_report.txt        the latest drift report; audit.json = spot-check answers
"""
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .objects import OBJECT_NAMES
from .selflabel import (HELD, LABELS_FILE, TRAIN, apply_answers, load_labels, load_questions,
                        pending_ids, walk_dirs)

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
        open_ids = {q for e in load_labels(d).values() for q in pending_ids(e)}
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

    from .drift import record_run, report
    record_run(models_dir, go, result, today=today)
    lines += report(frames_dir, models_dir)
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


# ------------------------------------------------------------------ original training photos

# Bump by one whenever a change to downloading, frame extraction, importing or auto-boxing
# changes the photos/boxes made, so the saved snapshot below is rebuilt once (task 29d).
PHOTO_PIPELINE_VERSION = 1
SNAPSHOT_SUBDIR = "training_set"  # in Drive claudeknows_models/
SNAPSHOT_TAR = "training_set.tar"
SNAPSHOT_MANIFEST = "manifest.json"
OWL_RUN_THRESHOLD = 0.3
OWL_FILL_THRESHOLD = 0.4
SKIP_FOOTAGE = ("walks",)  # walk videos are handled by the loop itself, not the original photos
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def photo_settings(cap=300, negatives=120, fps=2.0):
    """Everything that decides HOW the original photos, frames and boxes are made. A change
    to any value rebuilds the snapshot (RESEARCH.md 2026-11-23)."""
    from .autobox import OWL_MODEL
    from .detection_data import DETECT_SIDE
    from .frames import BLUR_THRESHOLD
    return {"cap": cap, "negatives": negatives, "owl_threshold": OWL_RUN_THRESHOLD,
            "owl_fill_threshold": OWL_FILL_THRESHOLD, "owl_model": OWL_MODEL, "fps": fps,
            "DETECT_SIDE": DETECT_SIDE, "BLUR_THRESHOLD": BLUR_THRESHOLD,
            "PHOTO_PIPELINE_VERSION": PHOTO_PIPELINE_VERSION}


def footage_list(photos_dir):
    """[[path, size], ...] of Alex's own photos/videos (not walks/), sorted. Names + sizes
    only: cheap to list in Drive, and a replaced or new file changes it."""
    root = Path(photos_dir) if photos_dir else None
    if root is None or not root.is_dir():
        return []
    out = []
    for f in root.rglob("*"):
        rel = f.relative_to(root)
        if f.is_file() and rel.parts[0] not in SKIP_FOOTAGE and not rel.name.startswith("."):
            out.append([rel.as_posix(), f.stat().st_size])
    return sorted(out)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip() or None
    except Exception:
        return None


def snapshot_id(snapshot_dir):
    """First 12 hex characters of the manifest's sha256, or None if there is no snapshot."""
    m = Path(snapshot_dir) / SNAPSHOT_MANIFEST
    return _sha256(m)[:12] if m.exists() else None


def _footage_change(old, new):
    old, new = dict(map(tuple, old)), dict(map(tuple, new))
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    parts = [f"{len(x)} {w}" for x, w in ((added, "new"), (removed, "removed"), (changed, "changed"))
             if x]
    example = (added + changed + removed)[0]
    return f"your own photos/videos changed ({', '.join(parts)}, e.g. {example})"


def snapshot_mismatch(manifest, settings, footage):
    """Why a saved snapshot can't be reused (plain language), or None if it can. Settings are
    compared key by key; a key an older manifest lacks counts as changed."""
    old = manifest.get("settings", {})
    changed = sorted(k for k in set(old) | set(settings) if old.get(k, ...) != settings.get(k, ...))
    if changed:
        return f"photo settings changed ({', '.join(changed)})"
    if manifest.get("footage", []) != footage:
        return _footage_change(manifest.get("footage", []), footage)
    return None


def _photo_counts(files):
    """{"web/keys": n, ...} images per source/object."""
    counts = Counter()
    for rel in files:
        parts = rel.split("/")
        if len(parts) >= 3 and rel.lower().endswith(IMAGE_EXTS):
            counts[f"{parts[0]}/{parts[1]}"] += 1
    return dict(sorted(counts.items()))


def write_snapshot(root, snapshot_dir, settings, footage, today=None):
    """Save every file under `root` as ONE uncompressed tar + manifest.json in `snapshot_dir`.
    Both are written under temporary names and renamed, so a disconnect never leaves half a
    snapshot (the manifest is renamed last). Returns the manifest."""
    import os
    import tarfile
    root, snap = Path(root), Path(snapshot_dir)
    snap.mkdir(parents=True, exist_ok=True)
    files = sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file())
    tmp_tar = snap / (SNAPSHOT_TAR + ".tmp")
    with tarfile.open(tmp_tar, "w") as tar:
        for rel in files:
            tar.add(root / rel, arcname=rel, recursive=False)
    manifest = {"date": (today or date.today()).isoformat(), "git_commit": _git_commit(),
                "settings": settings, "footage": footage, "counts": _photo_counts(files),
                "photos": sum(_photo_counts(files).values()),
                "files": {rel: _sha256(root / rel) for rel in files}}
    tmp_man = snap / (SNAPSHOT_MANIFEST + ".tmp")
    tmp_man.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp_tar, snap / SNAPSHOT_TAR)
    os.replace(tmp_man, snap / SNAPSHOT_MANIFEST)
    return manifest


def restore_snapshot(snapshot_dir, root, manifest):
    """Copy the tar to local disk, unpack it next to `root` and check every file's sha256.
    Only if all match does it replace `root`. Returns None if fine, else the reason."""
    import shutil
    import tarfile
    root, snap = Path(root), Path(snapshot_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    local = root.parent / f".{root.name}.snapshot.tar"
    staging = root.parent / f".{root.name}.restoring"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if not (snap / SNAPSHOT_TAR).exists():
            return "the saved photo file is missing"
        shutil.copyfile(snap / SNAPSHOT_TAR, local)  # one big Drive read, then local disk only
        try:
            with tarfile.open(local) as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(staging, filter="data")
                else:  # Python < 3.12
                    tar.extractall(staging)
        except (tarfile.TarError, OSError) as e:
            return f"the saved photo file could not be unpacked ({e})"
        bad = [rel for rel, h in sorted(manifest.get("files", {}).items())
               if not (staging / rel).is_file() or _sha256(staging / rel) != h]
        if bad:
            return f"{len(bad)} saved file(s) missing or damaged, e.g. {bad[0]}"
        shutil.rmtree(root, ignore_errors=True)
        staging.rename(root)
        return None
    finally:
        local.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def build_training_photos(root, photos_dir=None, cap=300, negatives=120, fps=2.0, run=None):
    """Make the original training photos from scratch, the same steps as the spatial
    notebook: Open Images download, Alex's Drive footage, OWLv2 boxes (slow)."""
    import subprocess
    import sys
    run = run or (lambda cmd: subprocess.run(cmd, check=True))
    py = [sys.executable, "-m"]
    run(py + ["claudeknows.detection_data", "download", "--out", str(root), "--cap", str(cap),
              "--negatives", str(negatives)])
    if photos_dir and Path(photos_dir).is_dir():
        run(py + ["claudeknows.detection_data", "import", str(photos_dir), "--root", str(root),
                  "--fps", str(fps)])
    run(py + ["claudeknows.autobox", "run", str(root), "--threshold", str(OWL_RUN_THRESHOLD)])
    run(py + ["claudeknows.autobox", "fill", str(root), "--threshold", str(OWL_FILL_THRESHOLD)])


def prepare_training_photos(root, photos_dir=None, snapshot_dir=None, cap=300, negatives=120,
                            fps=2.0, run=None, today=None, log=print):
    """Put the original training photos in `root` (Colab's temporary disk).

    With `snapshot_dir` (Drive `claudeknows_models/training_set/`, task 29d): reuse the saved
    snapshot if its settings and Alex's own-footage list still match and every file's hash
    checks out; otherwise build as before and save a new snapshot. So every retrain trains
    on the SAME original photos, and skips re-downloading and OWLv2. Without `snapshot_dir`
    it always builds. Only called when a retrain will really happen.
    Returns {"snapshot": id or None, "reused": bool, "photos": n, "message": str}."""
    settings = photo_settings(cap, negatives, fps)
    if snapshot_dir is None:
        build_training_photos(root, photos_dir, cap, negatives, fps, run)
        return {"snapshot": None, "reused": False, "photos": None,
                "message": "training photos: built fresh (no snapshot folder given)"}
    snap = Path(snapshot_dir)
    footage = footage_list(photos_dir)
    manifest = _read(snap / SNAPSHOT_MANIFEST, None)
    if manifest is None:
        why = "no saved snapshot yet"
    else:
        why = snapshot_mismatch(manifest, settings, footage)
        if why is None:
            why = restore_snapshot(snap, root, manifest)
            if why is None:
                sid = snapshot_id(snap)
                msg = f"training photos: snapshot {sid} ({manifest.get('photos', 0)} photos) reused"
                log(msg)
                return {"snapshot": sid, "reused": True, "photos": manifest.get("photos", 0),
                        "message": msg}
    log(f"training photos: building ({why}); this takes a while")
    build_training_photos(root, photos_dir, cap, negatives, fps, run)
    manifest = write_snapshot(root, snap, settings, footage, today)
    sid = snapshot_id(snap)
    msg = f"training photos rebuilt: {why}; new snapshot {sid} ({manifest['photos']} photos) saved"
    log(msg)
    return {"snapshot": sid, "reused": False, "photos": manifest["photos"], "message": msg}


def colab_retrain_fn(root, models_dir, frames_dir, photos_dir=None, gold_dir=None, epochs=15,
                     workers=2, noise_floor=True, snapshot_dir=None, log=print):
    """retrain_fn for Colab: put the original photos in `root` (from the Drive snapshot when
    possible, task 29d), measure any missing noise-floor seeds (only with a gold set; again
    whenever the snapshot changes), then train + gate one candidate on the original photos +
    approved walk frames. The gold set lives in `gold_dir` (Drive)."""
    from argparse import Namespace
    from . import retrain
    from .selflabel import self_samples
    snapshot_dir = snapshot_dir or Path(models_dir) / SNAPSHOT_SUBDIR

    def run():
        photos = prepare_training_photos(root, photos_dir, snapshot_dir, log=log)
        sid = photos["snapshot"]
        a = Namespace(root=root, gold_dir=gold_dir, epochs=epochs, batch_size=4, patience=4,
                      workers=workers, max_batches=None)
        base, gsamps, gold, kwargs = retrain._data(a, log)
        if noise_floor and gsamps:
            missing = retrain.noise_progress(models_dir, retrain.recipe_of(kwargs, training_set=sid),
                                             log=log)[2]
            if missing:
                log(f"measuring the noise floor first: {len(missing)} short training(s), saved one by one")
                retrain.measure_noise_floor(base, gsamps, models_dir, gold_manifest=gold,
                                            train_kwargs=kwargs, training_set=sid, log=log)
        extra = self_samples(frames_dir, gold, log)
        current = retrain.load_versions(models_dir).get("current")
        result = retrain.retrain(base, extra, gsamps, models_dir, gold_manifest=gold,
                                 train_kwargs=kwargs, training_set=sid, log=log)
        notes = [photos["message"]]
        cur = next((e for e in retrain.load_versions(models_dir)["all"] if e["name"] == current), None)
        if retrain.current_path(models_dir) and (cur is None or cur.get("training_set") != sid) \
                and result.get("decision") is not None:
            notes.append("note: the current model was trained on a different set of original photos, "
                         "so this comparison is not yet like-for-like")
        result["summary"] = notes + result.get("summary", [])
        result["training_set"] = sid
        return result
    return run
