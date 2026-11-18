"""Phase 2, step 3: auto-label walk frames (task 26).

Two independent labellers look at every walk frame: the current detector and
OWLv2. A box becomes a training label only when they agree, and the object is
seen steadily over neighbouring frames. Anything else that looks confident
becomes a yes/no question for Alex (a short review batch, at most 20).

Rules, in order (RESEARCH.md 2026-10-21, 2026-11-05, 2026-11-06):
1. Agreement: a detector box and an OWLv2 box with the same label, IoU >= 0.5,
   both above their own cut-off (detector: per-object cut-off from the
   checkpoint, task 23m; OWLv2: `autobox.THRESHOLD`). The label box is the mean
   of the two.
2. Steadiness: an agreed box is kept only if a same-label agreed box with
   IoU > 0.3 is in at least 2 of the 4 nearest frames of the same walk. If it
   fails: a question for Alex if both scores are high (>= `STEADY_REVIEW`),
   else dropped.
3. One labeller only (a confident box the other did not confirm): a question.
4. No half-labelled frames: a frame is used for training only if EVERY
   confident box was agreed+steady or answered by Alex. Otherwise it is held
   back (a real object left unlabelled would be taught as "background").
5. Questions are sorted by how much the current detector disagrees with the
   proposed boxes (its box-head loss on them, highest first) and capped at 20.
   The number only orders the list; nothing is deleted because of it.

6. Spot checks (task 29, RESEARCH.md 2026-11-12): `AUDIT_PER_WALK` agreed, steady
   boxes are picked at random (seeded by the walk name) and mixed into the review
   batch as ordinary-looking questions (`"why": "audit"`). They count toward the cap
   of 20 but never hold a frame back; "no" removes that box from the frame's labels.
   Their answers measure how often the ACCEPTED auto-labels are wrong (`drift.py`).

Per walk, the per-object counts of agreed / detector-only / OWLv2-only boxes
(before the steadiness rule) are saved in `walkstats.json`; the drift report turns
them into an agreement rate per object (RESEARCH.md 2026-11-15).

Frames where nobody found anything confident are not used (they might still
hide an object both missed). Self-labelled frames may make up at most
`MAX_SELF_SHARE` (30%) of the training photos (`cap_self_share`).

Output per walk folder: `selflabels.json` (frame -> status + boxes) and
`review.json` (the questions). Both live next to the frames on Colab/Drive and
are never committed (the repo is public).

    python -m claudeknows.selflabel label data/walks --checkpoint detector.pt   # Colab (OWLv2)
    python -m claudeknows.selflabel questions data/walks
    python -m claudeknows.selflabel answer data/walks --yes walk1:3 --no walk1:5
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from .autobox import THRESHOLD as OWL_THRESHOLD
from .autobox import cutoff, iou
from .detection_data import FOLDERS, TOGETHER, DetSample
from .objects import OBJECT_NAMES

AGREE_IOU = 0.5
STEADY_IOU = 0.3
STEADY_NEIGHBOURS = 4  # look at the 4 nearest frames ...
STEADY_MIN = 2  # ... and want the object in at least 2 of them
STEADY_REVIEW = 0.6  # an unsteady agreed box is asked about only if both scores reach this
MAX_QUESTIONS = 20
MAX_SELF_SHARE = 0.3
LOW_SCORE = 0.05  # detector boxes below this are not even looked at
LABELS_FILE = "selflabels.json"
REVIEW_FILE = "review.json"
STATS_FILE = "walkstats.json"
AUDIT_PER_WALK = 3
AUDIT = "audit"  # `why` of a spot-check question
SELF = "self"  # `by` of a self-labelled box

TRAIN, HELD, EMPTY = "train", "held", "empty"


def _round(box):
    return [round(float(v), 4) for v in box]


def _mean_box(a, b):
    return tuple((x + y) / 2 for x, y in zip(a, b))


def agree(det_found, owl_found, det_cut, owl_cut=OWL_THRESHOLD, iou_thresh=AGREE_IOU):
    """Match one frame's boxes from the two labellers.

    det_found: [{"label", "score", "box"}] (as `detector.detect`), owl_found:
    [(label, score, box)] (as `autobox.owl_detector`). `det_cut(label)` is the
    detector's cut-off. Returns (agreed, single): agreed = [{"label", "box",
    "score" (the lower of the two)}], single = confident boxes only one labeller
    saw, [{"label", "box", "score", "seen_by"}]. Low-score boxes are ignored.
    """
    dets = [d for d in det_found if d["score"] >= det_cut(d["label"])]
    owls = [{"label": l, "score": s, "box": tuple(b)} for l, s, b in owl_found
            if s >= cutoff(owl_cut, l)]
    pairs = sorted(((iou(d["box"], o["box"]), i, j) for i, d in enumerate(dets)
                    for j, o in enumerate(owls) if d["label"] == o["label"]), reverse=True)
    used_d, used_o, agreed = set(), set(), []
    for v, i, j in pairs:
        if v < iou_thresh or i in used_d or j in used_o:
            continue
        used_d.add(i)
        used_o.add(j)
        d, o = dets[i], owls[j]
        agreed.append({"label": d["label"], "box": _mean_box(d["box"], o["box"]),
                       "score": min(d["score"], o["score"])})
    single = ([{**d, "box": tuple(d["box"]), "seen_by": "detector"}
               for i, d in enumerate(dets) if i not in used_d]
              + [{**o, "seen_by": "owlv2"} for j, o in enumerate(owls) if j not in used_o])
    return agreed, single


def steady(frames_agreed, k=STEADY_NEIGHBOURS, need=STEADY_MIN, iou_thresh=STEADY_IOU):
    """Frame-to-frame check. frames_agreed: one list of agreed boxes per frame, in
    walk order. Returns a list of booleans per frame/box: True if the same label
    (IoU > iou_thresh) is in at least `need` of the `k` nearest other frames.
    A very short walk (fewer than `need` other frames) needs all of them."""
    n = len(frames_agreed)
    out = []
    for i, boxes in enumerate(frames_agreed):
        near = sorted((j for j in range(n) if j != i), key=lambda j: (abs(j - i), j))[:k]
        want = min(need, len(near))
        flags = []
        for b in boxes:
            hits = sum(any(o["label"] == b["label"] and iou(o["box"], b["box"]) > iou_thresh
                           for o in frames_agreed[j]) for j in near)
            flags.append(hits >= want and want > 0)
        out.append(flags)
    return out


def review_loss(model, image, boxes, classes=OBJECT_NAMES):
    """How much the current detector disagrees with proposed `boxes` (fractions).

    The proposed boxes are fed to the detector's box head as if they were its own
    proposals, and we take its usual `loss_classifier + loss_box_reg` on them:
    "is this box that object?" plus "would you move this box?". Bigger = more
    suspicious. (Taking the loss in train mode instead, as first planned, mostly
    measures the random background samples the detector draws, so it ranked
    boxes badly in tests.) Eval mode, no gradients, deterministic."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    img = image.convert("RGB")
    w, h = img.size
    device = next(model.parameters()).device
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            images, _ = model.transform([t])
            th, tw = images.image_sizes[0]
            px = torch.tensor([[b["box"][0] * tw, b["box"][1] * th, b["box"][2] * tw, b["box"][3] * th]
                               for b in boxes], dtype=torch.float32, device=device).reshape(-1, 4)
            labels = torch.tensor([classes.index(b["label"]) + 1 for b in boxes], device=device)
            feats = model.backbone(images.tensors)
            if isinstance(feats, torch.Tensor):  # single-map backbone (as torchvision does)
                feats = {"0": feats}
            rh = model.roi_heads
            x = rh.box_head(rh.box_roi_pool(feats, [px], images.image_sizes))
            logits, reg = rh.box_predictor(x)
            cls_loss = F.cross_entropy(logits, labels)
            reg = reg.reshape(len(px), -1, 4)[torch.arange(len(px)), labels]
            # the target box IS the proposal, so the ideal regression is all zeros
            box_loss = F.smooth_l1_loss(reg, torch.zeros_like(reg), beta=1 / 9, reduction="sum") / len(px)
    finally:
        model.train(was_training)
    return float(cls_loss + box_loss)


def _frames(walk_dir):
    return sorted(p for p in Path(walk_dir).glob("*.jpg"))


def label_walk(walk_dir, det, owl, det_cut, owl_cut=OWL_THRESHOLD, loss_fn=None,
               max_questions=MAX_QUESTIONS, audit_per_walk=AUDIT_PER_WALK, log=print):
    """Auto-label one walk folder (frames from `walks.import_walks`).

    det(image) -> detector boxes (low cut-off, e.g. `detect(model, im, min_score=LOW_SCORE)`),
    owl(image, labels) -> OWLv2 boxes, det_cut(label) -> detector cut-off,
    loss_fn(image, boxes) -> disagreement number for ordering questions (None = no ordering).
    audit_per_walk: agreed, steady boxes to spot-check (task 29; 0 = none).
    Writes `selflabels.json`, `review.json` and `walkstats.json`; returns the counts per rule.
    """
    walk_dir = Path(walk_dir)
    frames = _frames(walk_dir)
    per_frame = []
    for p in frames:
        with Image.open(p) as im:
            im = im.convert("RGB")
            per_frame.append(agree(det(im), owl(im, list(OBJECT_NAMES)), det_cut, owl_cut))
    flags = steady([a for a, _ in per_frame])
    per_object = {c: {"agreed": 0, "detector_only": 0, "owlv2_only": 0} for c in OBJECT_NAMES}
    for agreed_boxes, single in per_frame:  # before the steadiness rule
        for b in agreed_boxes:
            per_object.setdefault(b["label"], {"agreed": 0, "detector_only": 0, "owlv2_only": 0})
            per_object[b["label"]]["agreed"] += 1
        for b in single:
            per_object.setdefault(b["label"], {"agreed": 0, "detector_only": 0, "owlv2_only": 0})
            per_object[b["label"]][f"{b['seen_by']}_only"] += 1

    labels, questions = {}, []
    stats = {"frames": len(frames), "agreed": 0, "unsteady_dropped": 0, "unsteady_asked": 0,
             "single_asked": 0, "questions_over_cap": 0}
    for p, (agreed, single), ok in zip(frames, per_frame, flags):
        keep, open_q = [], []
        for b, good in zip(agreed, ok):
            if good:
                keep.append(b)
                stats["agreed"] += 1
            elif b["score"] >= STEADY_REVIEW:
                open_q.append({**b, "why": "not steady over neighbouring frames"})
                stats["unsteady_asked"] += 1
            else:
                open_q.append({**b, "why": "dropped"})  # confident yet unresolved: holds the frame
                stats["unsteady_dropped"] += 1
        for s in single:
            open_q.append({"label": s["label"], "box": s["box"], "score": s["score"],
                           "why": f"only {s['seen_by']} saw it"})
            stats["single_asked"] += 1
        entry = {"boxes": [{"label": b["label"], "box": _round(b["box"]),
                            "score": round(b["score"], 3)} for b in keep],
                 "open": [], "answered": []}
        for q in open_q:
            if q["why"] == "dropped":
                entry["open"].append(None)  # marker: never asked, frame stays held back
                continue
            qid = f"{walk_dir.name}:{len(questions)}"
            entry["open"].append(qid)
            questions.append({"id": qid, "frame": p.name, "label": q["label"], "box": _round(q["box"]),
                              "score": round(q["score"], 3), "why": q["why"]})
        labels[p.name] = entry

    # spot checks: a seeded random pick of accepted boxes, the same on every rerun
    pool = [(name, i) for name, e in sorted(labels.items()) for i in range(len(e["boxes"]))]
    rng = random.Random(f"audit:{walk_dir.name}")
    picks = sorted(rng.sample(pool, min(audit_per_walk, len(pool), max_questions)))
    audits = []
    for k, (name, i) in enumerate(picks):
        b = labels[name]["boxes"][i]
        qid = f"{walk_dir.name}:a{k}"
        labels[name].setdefault("audit", []).append(qid)
        audits.append({"id": qid, "frame": name, "label": b["label"], "box": b["box"],
                       "score": b["score"], "why": AUDIT})

    if loss_fn is not None:
        for q in questions:
            with Image.open(walk_dir / q["frame"]) as im:
                q["disagreement"] = round(loss_fn(im.convert("RGB"), [q]), 4)
        questions.sort(key=lambda q: -q["disagreement"])
    room = max_questions - len(audits)
    stats["questions_over_cap"] = max(0, len(questions) - room)
    questions = questions[:room]  # the rest stay unanswered: their frames are held back
    for q in audits:  # mixed in at random places, not ordered by disagreement
        questions.insert(rng.randint(0, len(questions)), q)
    stats["audits"] = len(audits)
    stats["per_object"] = per_object
    for entry in labels.values():
        entry["status"] = _status(entry)
    stats.update(_status_counts(labels))
    (walk_dir / LABELS_FILE).write_text(json.dumps(labels, indent=0, sort_keys=True))
    (walk_dir / REVIEW_FILE).write_text(json.dumps(questions, indent=1))
    (walk_dir / STATS_FILE).write_text(json.dumps(stats, indent=1, sort_keys=True))
    log(f"{walk_dir.name}: {stats['frames']} frames, {stats['agreed']} agreed steady boxes, "
        f"{stats['unsteady_dropped']} unsteady dropped, {stats['unsteady_asked']} unsteady + "
        f"{stats['single_asked']} one-labeller boxes to ask about ({len(questions)} questions"
        + (f", {stats['questions_over_cap']} over the cap of {max_questions}" if stats["questions_over_cap"] else "")
        + (f", {len(audits)} of them spot checks" if audits else "")
        + f"); frames: {stats[TRAIN]} for training, {stats[HELD]} held back, {stats[EMPTY]} empty")
    return stats


def _status(entry):
    if entry["open"]:
        return HELD
    return TRAIN if entry["boxes"] else EMPTY


def _status_counts(labels):
    counts = {TRAIN: 0, HELD: 0, EMPTY: 0}
    for e in labels.values():
        counts[e["status"]] += 1
    return counts


def load_labels(walk_dir):
    f = Path(walk_dir) / LABELS_FILE
    return json.loads(f.read_text()) if f.exists() else {}


def load_stats(walk_dir):
    f = Path(walk_dir) / STATS_FILE
    return json.loads(f.read_text()) if f.exists() else None


def pending_ids(entry):
    """Question ids of one frame still waiting for a yes/no (review + spot checks)."""
    return [q for q in entry["open"] if q] + list(entry.get("audit", []))


def load_questions(walk_dir):
    f = Path(walk_dir) / REVIEW_FILE
    return json.loads(f.read_text()) if f.exists() else []


def apply_answers(walk_dir, answers, log=print):
    """Record Alex's yes/no answers: {question id: True/False}. "yes" adds the box
    as a label, "no" removes the question (that spot is not the object). A frame
    becomes trainable once all its open questions are answered. A spot-check
    (audit) question never holds a frame: "yes" keeps its box, "no" removes it.
    Returns status counts."""
    walk_dir = Path(walk_dir)
    labels = load_labels(walk_dir)
    questions = {q["id"]: q for q in load_questions(walk_dir)}
    for qid, yes in answers.items():
        q = questions.get(qid)
        if q is None:
            log(f"unknown question {qid}, skipped")
            continue
        entry = labels[q["frame"]]
        if qid in entry.get("audit", []):
            entry["audit"].remove(qid)
            entry["answered"].append({"id": qid, "yes": bool(yes), "audit": True,
                                      "label": q["label"]})
            if not yes:
                entry["boxes"] = [b for b in entry["boxes"]
                                  if not (b["label"] == q["label"] and list(b["box"]) == list(q["box"]))]
            continue
        if qid not in entry["open"]:
            continue
        entry["open"].remove(qid)
        entry["answered"].append({"id": qid, "yes": bool(yes), "label": q["label"]})
        if yes:
            entry["boxes"].append({"label": q["label"], "box": q["box"], "score": q["score"],
                                   "by": "alex"})
    for entry in labels.values():
        entry["status"] = _status(entry)
    (walk_dir / LABELS_FILE).write_text(json.dumps(labels, indent=0, sort_keys=True))
    return _status_counts(labels)


def walk_dirs(root):
    """Walk folders under `root` that have been auto-labelled."""
    root = Path(root)
    return sorted(d for d in root.iterdir() if (d / LABELS_FILE).exists()) if root.is_dir() else []


def self_samples(root, gold=None, log=print):
    """Training samples from every labelled walk (status "train" only), source "walk".
    `gold` (a gold manifest) leaves out any gold walk."""
    out = []
    for d in walk_dirs(root):
        for name, e in sorted(load_labels(d).items()):
            if e["status"] != TRAIN or not (d / name).exists():
                continue
            boxes = tuple((b["label"], tuple(b["box"])) for b in e["boxes"])
            out.append(DetSample(d / name, FOLDERS.index(TOGETHER), "walk", d.name, TOGETHER,
                                 boxes, SELF))
    if gold is not None:
        from .gold import remove_gold
        out = remove_gold(out, gold, log)
    return out


def _rarest_groups(extra, seed):
    """Self-labelled frames grouped by the rarest object among their boxes (rarity =
    self-labelled boxes of that object in `extra`), rarest group first, each shuffled."""
    totals = Counter(l for s in extra for l, _ in getattr(s, "boxes", ()))
    groups = {}
    for s in extra:
        labels = {l for l, _ in getattr(s, "boxes", ())}
        key = min(labels, key=lambda l: (totals[l], l)) if labels else ""
        groups.setdefault(key, []).append(s)
    rng = random.Random(seed)
    out = []
    for key in sorted(groups, key=lambda k: (totals.get(k, 0), k)):
        g = list(groups[key])
        rng.shuffle(g)
        out.append(g)
    return out


def self_box_usage(extra, picked):
    """Plain-language line: self-labelled boxes used per object, e.g.
    "self-labelled boxes used: keys 40 of 40, water_bottle 150 of 300, shoes 150 of 620"."""
    have = Counter(l for s in extra for l, _ in s.boxes)
    used = Counter(l for s in picked for l, _ in s.boxes)
    return "self-labelled boxes used: " + ", ".join(
        f"{c} {used.get(c, 0)} of {have.get(c, 0)}" for c in OBJECT_NAMES)


def cap_self_share(base, extra, max_share=MAX_SELF_SHARE, seed=0, log=print):
    """base + a pick of `extra` (self-labelled) so that self-labelled photos are at most
    `max_share` of the result.

    When frames must be dropped, the rarest object comes first (task 29b, RESEARCH.md
    2026-11-15): each frame joins the group of the rarest object it shows, each group is
    shuffled with the seed, and frames are taken round-robin from the groups, rarest group
    first. Otherwise keys (which the labellers find least often) would keep being crowded
    out by shoe and bottle frames, walk after walk. Under the limit everything is kept."""
    limit = int(max_share / (1 - max_share) * len(base)) if max_share < 1 else len(extra)
    picked = list(extra)
    if len(picked) > limit:
        groups = _rarest_groups(extra, seed)
        picked, depth = [], 0
        while len(picked) < limit:
            for g in groups:
                if depth < len(g) and len(picked) < limit:
                    picked.append(g[depth])
            depth += 1
        log(f"self-labelled frames capped at {limit} of {len(extra)} "
            f"({int(max_share * 100)}% of training), rarest object first")
    if extra and hasattr(extra[0], "boxes"):
        log(self_box_usage(extra, picked))
    return list(base) + picked


def _parse_ids(values):
    return [v for item in values or [] for v in item.split(",") if v]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Auto-label walk frames (detector + OWLv2 must agree)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label", help="label every walk folder (Colab: needs OWLv2)")
    lb.add_argument("root", help="folder holding one sub-folder of frames per walk")
    lb.add_argument("--checkpoint", required=True, help="current detector.pt")
    lb.add_argument("--redo", action="store_true", help="also re-label walks done before")
    qs = sub.add_parser("questions", help="list the review questions")
    qs.add_argument("root")
    an = sub.add_parser("answer", help="record yes/no answers (ids like walk1:3)")
    an.add_argument("root")
    an.add_argument("--yes", nargs="*", default=[])
    an.add_argument("--no", nargs="*", default=[])
    a = ap.parse_args(argv)
    root = Path(a.root)

    if a.cmd == "label":
        from .autobox import owl_detector
        from .detector import _cut, detect, load_detector
        model, classes = load_detector(a.checkpoint)
        owl = owl_detector()
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if (d / LABELS_FILE).exists() and not a.redo:
                continue
            label_walk(d, lambda im: detect(model, im, classes, min_score=LOW_SCORE), owl,
                       lambda l: _cut(model, l, None),
                       loss_fn=lambda im, boxes: review_loss(model, im, boxes, classes))
    elif a.cmd == "questions":
        for d in walk_dirs(root):
            for q in load_questions(d):
                if any(q["id"] in pending_ids(e) for e in load_labels(d).values()):
                    print(f"{q['id']}  {q['frame']}: is this a {q['label']}? ({q['why']}, "
                          f"score {q['score']})")
    else:
        answers = {**{i: True for i in _parse_ids(a.yes)}, **{i: False for i in _parse_ids(a.no)}}
        for d in walk_dirs(root):
            mine = {k: v for k, v in answers.items() if k.split(":")[0] == d.name}
            if mine:
                print(d.name, apply_answers(d, mine))


if __name__ == "__main__":
    main()
