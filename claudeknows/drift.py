"""Drift report for the learning loop (roadmap task 29, Phase 2).

A self-learning loop can slowly teach the model its own mistakes. This report
watches for that across loop runs and writes plain-language lines (and warnings)
into the loop summary. It never blocks anything; the gold-set gate (task 27) is
still the only thing that decides whether a model is promoted.

What it tracks (everything lives in Drive `claudeknows_models/`, never in git):
  1. Gold-set scores of the current model and of each candidate, per loop run
     (`drift_history.json`, one row per run). Warns if the current model scores
     lower on the SAME gold set than it did in an earlier run.
  2. Self-labelled boxes per object per walk. Warns if an object that earlier
     walks had stops showing up in the last 3 walks.
  3. Spot checks of ACCEPTED auto-labels (RESEARCH.md 2026-11-12): a few agreed,
     steady boxes per walk are asked about like any other question. Their "no"
     share is the real error rate of the auto-labels, with a 95% Wilson range.
     Warns when the low end of that range is above 5% (clearly worse than 1 in 20);
     with fewer than 20 checks it says "too few checks yet". The running log is
     `audit.json` (walk, object, yes/no).
     The "no" rate of the ordinary review questions is reported separately: it
     describes the UNCERTAIN boxes, not the accepted ones.
  4. Agreement rate per object per walk (RESEARCH.md 2026-11-15): agreed /
     (agreed + detector-only + OWLv2-only), counted before the steadiness rule.
     A label-free early warning that a new room or lighting is harder. Warns only
     when the last 3 walks are each more than 15 points below the median of the
     earlier walks (walks with at least 10 boxes of that object; at least 5 such
     walks needed in total).
  5. Labeller check on the gold set (task 29f, RESEARCH.md 2026-11-19 and 2026-11-26):
     precision and recall of the current detector and of the "both agree" rule against
     the gold boxes, per object, with 95% Wilson ranges. OWLv2 is NOT scored against
     gold (the gold boxes ARE OWLv2's boxes, so it would look perfect); instead its
     real hit rate comes from Alex's Yes/No answers in the gold-set notebook. Report
     only: it never changes a cut-off, the agreement rule or the gate.

    python -m claudeknows.drift --walks DRIVE/claudeknows_models/walk_frames --models-dir DRIVE/claudeknows_models
"""
import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

from .autobox import REJECTED, THRESHOLD as OWL_THRESHOLD, cutoff, iou, nms
from .objects import OBJECT_NAMES
from .selflabel import AGREE_IOU, agree, load_labels, load_stats, walk_dirs

HISTORY_FILE = "drift_history.json"
AUDIT_LOG = "audit.json"
REPORT_FILE = "drift_report.txt"
MIN_AUDITS = 20  # fewer spot checks than this -> "too few checks yet"
AUDIT_WARN = 0.05  # warn when the low end of the wrong-share range is above this
AGREE_MIN_BOXES = 10  # a walk counts for an object's agreement rate only with this many boxes
AGREE_MIN_WALKS = 5  # ... and the rule needs this many such walks
AGREE_RECENT = 3  # the last 3 walks ...
AGREE_DROP = 0.15  # ... each more than 15 points below the earlier median
GOLD_DROP = 0.02  # current model's gold mAP lower than an earlier run's by more than 2 points
MISSING_RECENT = 3  # an object absent from the last 3 walks' self-labels (but seen before)
LABELLER_FILE = "labeller_check.json"  # task 29f: cached numbers + the lines shown in the report
GOLD_ANSWERS = "gold_answers.json"  # Alex's Yes/No clicks in the gold-set notebook
MIN_GOLD_BOXES = 10  # fewer gold boxes of an object -> "too few gold boxes"
OWL_SUGGEST = 0.7  # suggest looking at OWLv2 when it got fewer than 70% of a folder's photos right
OWL_SUGGEST_MIN = 5  # ... with at least this many answers


def _read(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ spot checks

def wilson(k, n, z=1.96):
    """(share, low, high): k of n with a 95% Wilson score range (good for small n and k=0)."""
    if n == 0:
        return None
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    den = 1 + z * z / n
    return p, max(0.0, (centre - half) / den), min(1.0, (centre + half) / den)


def answered(walks_root):
    """Every yes/no answer so far, one row per question: [{"walk", "frame", "object", "yes",
    "audit", "track"}]. A track question (task 29c) is saved in each of its frames but
    counted once."""
    rows = []
    for d in walk_dirs(walks_root):
        seen = set()
        for frame, e in sorted(load_labels(d).items()):
            for a in e.get("answered", []):
                if a.get("id") is not None:
                    if a["id"] in seen:
                        continue
                    seen.add(a["id"])
                rows.append({"walk": d.name, "frame": frame, "object": a.get("label"),
                             "yes": bool(a["yes"]), "audit": bool(a.get("audit")),
                             "track": bool(a.get("track"))})
    return rows


def write_audit_log(walks_root, models_dir):
    """Rewrite Drive `audit.json` from the walk folders (spot-check answers only)."""
    rows = [{"walk": r["walk"], "object": r["object"], "yes": r["yes"]}
            for r in answered(walks_root) if r["audit"]]
    _write(Path(models_dir) / AUDIT_LOG, rows)
    return rows


def audit_lines(rows, classes=OBJECT_NAMES, min_n=MIN_AUDITS, warn=AUDIT_WARN):
    """(lines, warnings) for the spot checks of accepted auto-labels."""
    lines, warns = [], []
    groups = [("overall", rows)] + [(c, [r for r in rows if r["object"] == c]) for c in classes]
    for name, rs in groups:
        n, wrong = len(rs), sum(not r["yes"] for r in rs)
        if n == 0:
            lines.append(f"  {name}: no spot checks answered yet")
            continue
        p, lo, hi = wilson(wrong, n)
        line = f"  {name}: {wrong} of {n} spot-checked auto-labels wrong ({_pct(p)}, range {_pct(lo)}-{_pct(hi)})"
        if n < min_n:
            line += "; too few checks yet"
        elif lo > warn:
            warns.append(f"WARNING: {name}: auto-labels are wrong more often than 1 in 20 "
                         f"({wrong} of {n} spot checks, at least {_pct(lo)}); the loop may be "
                         "teaching the model its own mistakes")
        lines.append(line)
    return lines, warns


def review_no_line(rows):
    rs = [r for r in rows if not r["audit"]]
    if not rs:
        return "review questions: none answered yet"
    no = sum(not r["yes"] for r in rs)
    line = (f"review questions (the uncertain boxes): {no} of {len(rs)} answered 'no' "
            f"({_pct(no / len(rs))})")
    tr = [r for r in rs if r.get("track")]
    if tr:  # one wrong "yes" here adds several boxes, so watch it on its own
        tno = sum(not r["yes"] for r in tr)
        line += (f"; questions covering several frames: {tno} of {len(tr)} 'no' "
                 f"({_pct(tno / len(tr))})")
    return line


# ------------------------------------------------------------------ agreement rate

def agreement_series(walks_root, classes=OBJECT_NAMES):
    """{object: [(walk, rate, boxes)]} in walk order, from each walk's `walkstats.json`."""
    out = {c: [] for c in classes}
    for d in walk_dirs(walks_root):
        s = load_stats(d)
        per = (s or {}).get("per_object") or {}
        for c in classes:
            v = per.get(c)
            if not v:
                continue
            n = v["agreed"] + v["detector_only"] + v["owlv2_only"]
            if n:
                out[c].append((d.name, v["agreed"] / n, n))
    return out


def agreement_warning(name, series, min_boxes=AGREE_MIN_BOXES, min_walks=AGREE_MIN_WALKS,
                      recent=AGREE_RECENT, drop=AGREE_DROP):
    """(note or None, warning or None) for one object's [(walk, rate, boxes)]."""
    import numpy as np
    usable = [(w, r) for w, r, n in series if n >= min_boxes]
    if len(usable) < min_walks:
        return f"  {name}: too few walks yet ({len(usable)} with {min_boxes}+ boxes)", None
    before = float(np.median([r for _, r in usable[:-recent]]))
    last = [r for _, r in usable[-recent:]]
    note = f"  {name}: {_pct(np.mean(last))} on the last {recent} walks, {_pct(before)} before"
    if all(r < before - drop for r in last):
        return note, (f"WARNING: {name}: the current model and OWLv2 agree less often on recent walks "
                      f"({_pct(np.mean(last))} vs {_pct(before)} before): new rooms or lighting may be "
                      "harder; the gold score and spot checks are the real test")
    return note, None


# ------------------------------------------------------------------ label counts per walk

def label_counts(walks_root, classes=OBJECT_NAMES):
    """[(walk, {object: self-labelled boxes in trainable frames})] in walk order."""
    out = []
    for d in walk_dirs(walks_root):
        c = {k: 0 for k in classes}
        for e in load_labels(d).values():
            if e["status"] == "train":
                for b in e["boxes"]:
                    c[b["label"]] = c.get(b["label"], 0) + 1
        out.append((d.name, c))
    return out


def missing_warnings(counts, classes=OBJECT_NAMES, recent=MISSING_RECENT):
    if len(counts) <= recent:
        return []
    warns = []
    for c in classes:
        earlier = sum(x[c] for _, x in counts[:-recent])
        last = sum(x[c] for _, x in counts[-recent:])
        if earlier and not last:
            warns.append(f"WARNING: {c}: no new self-labelled boxes in the last {recent} walks "
                         f"(earlier walks had {earlier}); film it again or check the review answers")
    return warns


# ------------------------------------------------------------------ gold scores across runs

def load_history(models_dir):
    return _read(Path(models_dir) / HISTORY_FILE, [])


def record_run(models_dir, retrained, result=None, today=None):
    """Append one row per loop run to `drift_history.json`."""
    result = result or {}
    gold = result.get("gold") or {}
    d = result.get("decision") or {}
    row = {"date": (today or date.today()).isoformat(), "retrained": bool(retrained),
           "candidate": result.get("name"), "promoted": bool(result.get("promoted")),
           "gold_created": result.get("gold_created"),
           "gold_current": gold.get("current"), "gold_candidate": gold.get("candidate"),
           "current_name": gold.get("current_name"),
           "difference": None if not d.get("overall") else round(d["overall"][0], 4)}
    rows = load_history(models_dir) + [row]
    _write(Path(models_dir) / HISTORY_FILE, rows)
    return rows


def gold_lines(history, drop=GOLD_DROP):
    """(lines, warnings) about gold-set scores across runs (same gold set only)."""
    scored = [r for r in history if r.get("gold_current") is not None]
    if not scored:
        return ["gold-set scores: none yet (no retrain has been compared on a gold set)"], []
    lines = ["gold-set mAP per retrain (current -> candidate):"]
    for r in scored[-5:]:
        cand = "" if r.get("gold_candidate") is None else f" -> {100 * r['gold_candidate']:.1f}%"
        verdict = "promoted" if r.get("promoted") else "kept current"
        lines.append(f"  {r['date']}: {100 * r['gold_current']:.1f}%{cand} ({verdict})")
    warns = []
    last = scored[-1]
    same = [r for r in scored[:-1] if r.get("gold_created") == last.get("gold_created")]
    if same:
        best = max(same, key=lambda r: r["gold_current"])
        if last["gold_current"] < best["gold_current"] - drop:
            warns.append(f"WARNING: the current model scores {100 * last['gold_current']:.1f}% on the gold "
                         f"set, lower than {100 * best['gold_current']:.1f}% on {best['date']}; was a "
                         "model rolled back or replaced by hand?")
    return lines, warns


# ------------------------------------------------------------------ labeller check (task 29f)

def match_gold(found, gold_boxes, classes=OBJECT_NAMES, iou_thresh=AGREE_IOU):
    """found: [(label, score, box)], gold_boxes: [(label, box)] of one photo.
    Highest score first, each proposal takes the unused same-label gold box it overlaps
    most (IoU >= iou_thresh); each gold box counts once. Returns {object: [hits, proposed, gold]}."""
    out = {c: [0, 0, 0] for c in classes}
    for label, _ in gold_boxes:
        out.setdefault(label, [0, 0, 0])[2] += 1
    used = set()
    for label, _, box in sorted(found, key=lambda f: -f[1]):
        out.setdefault(label, [0, 0, 0])[1] += 1
        best, best_j = iou_thresh, None
        for j, (gl, gb) in enumerate(gold_boxes):
            if j in used or gl != label:
                continue
            v = iou(box, gb)
            if v >= best:
                best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            out[label][0] += 1
    return out


def _add(total, part):
    for k, v in part.items():
        t = total.setdefault(k, [0, 0, 0])
        for i in range(3):
            t[i] += v[i]


def labeller_check(gold_samps, det, owl, det_cut, owl_cut=OWL_THRESHOLD, classes=OBJECT_NAMES):
    """Score the current detector and the agreement rule on the gold photos.

    det(image) -> [{"label", "score", "box"}] (as `detector.detect`; boxes below
    `det_cut(label)` are dropped here), owl(image, labels) -> [(label, score, box)]
    (as `autobox.owl_detector`; cut-off `owl_cut` then `autobox.nms`, as in auto-boxing).
    Agreement = `selflabel.agree`, before the steadiness check (one photo at a time).
    OWLv2 is only used inside the agreement rule: it drew the gold boxes, so scoring
    it against them would measure nothing. Nothing is changed (report only).
    Returns {"photos", "detector": {obj: [hits, proposed, gold]}, "agreement": {...}}."""
    from PIL import Image
    res = {"photos": 0, "detector": {c: [0, 0, 0] for c in classes},
           "agreement": {c: [0, 0, 0] for c in classes}}
    for s in gold_samps:
        with Image.open(s.path) as im:
            im = im.convert("RGB")
            dets = [d for d in det(im) if d["score"] >= det_cut(d["label"])]
            owls = nms([f for f in owl(im, list(classes)) if f[1] >= cutoff(owl_cut, f[0])])
        agreed, _ = agree(dets, owls, det_cut, owl_cut)
        gold_boxes = [(n, tuple(b)) for n, b in s.boxes]
        _add(res["detector"], match_gold([(d["label"], d["score"], d["box"]) for d in dets], gold_boxes, classes))
        _add(res["agreement"], match_gold([(a["label"], a["score"], a["box"]) for a in agreed],
                                          gold_boxes, classes))
        res["photos"] += 1
    return res


def _share(k, n):
    if not n:
        return "none"
    p, lo, hi = wilson(k, n)
    return f"{k} of {n} ({_pct(p)}, range {_pct(lo)}-{_pct(hi)})"


def labeller_lines(res, classes=OBJECT_NAMES, min_boxes=MIN_GOLD_BOXES):
    """Plain lines for the detector / agreement numbers of `labeller_check`."""
    lines = [f"labeller check on {res['photos']} gold photos (precision = boxes that were right, "
             "recall = gold boxes found):"]
    for c in classes:
        n_gold = res["detector"].get(c, [0, 0, 0])[2]
        if n_gold < min_boxes:
            lines.append(f"  {c}: too few gold boxes for {c} ({n_gold}; {min_boxes} needed)")
            continue
        for key, name in (("detector", "current detector"),
                          ("agreement", "both agree (before the steadiness check)")):
            hits, prop, gold = res[key].get(c, [0, 0, 0])
            lines.append(f"  {c}, {name}: precision {_share(hits, prop)}; recall {_share(hits, gold)}")
    lines.append("  note: detector and agreement numbers are measured on photos OWLv2 already got fully "
                 "right, so real recall on your rooms is probably lower")
    return lines


def _answer(v):
    """(ok, why, folder) from an old (true/false) or new ({"ok", "why", "folder"}) answer."""
    if isinstance(v, dict):
        return bool(v.get("ok")), v.get("why"), v.get("folder")
    return bool(v), None, None


def owl_footage(answers, manifest=None, own_root=None, classes=OBJECT_NAMES):
    """OWLv2 on Alex's footage, from his gold-set answers: per folder
    {"yes", "no", "wrong", "missing", "rejected"} (rejected = footage frames OWLv2 found
    nothing in, None when unknown). The folder of a photo comes from the answer itself
    (newer format), else the gold manifest (Yes photos), else `own_root/<folder>/boxes.json`,
    else "unknown". Keys starting with "_" (e.g. "_rejected") are not photos."""
    from .detection_data import FOLDERS, load_boxes
    by_manifest = {Path(e["path"]).name: e.get("folder") for e in (manifest or {}).get("images", [])}
    own = {}
    if own_root and Path(own_root).is_dir():
        for f in FOLDERS:
            for name, e in load_boxes(Path(own_root) / f).items():
                own.setdefault(f, {})[name] = e
    by_own = {name: f for f, entries in own.items() for name in entries}
    out = {}
    for name, v in sorted(answers.items()):
        if name.startswith("_"):
            continue
        ok, why, folder = _answer(v)
        folder = folder or by_manifest.get(name) or by_own.get(name) or "unknown"
        row = out.setdefault(folder, {"yes": 0, "no": 0, "wrong": 0, "missing": 0, "rejected": None})
        row["yes" if ok else "no"] += 1
        if not ok and why in ("wrong", "missing"):
            row[why] += 1
    saved_rejected = answers.get("_rejected") if isinstance(answers.get("_rejected"), dict) else {}
    for folder, row in out.items():
        if folder in own:
            row["rejected"] = sum(e.get("by") == REJECTED for e in own[folder].values())
        elif folder in saved_rejected:
            row["rejected"] = int(saved_rejected[folder])
    return out


def owl_footage_lines(rows, threshold=OWL_THRESHOLD):
    """(lines, suggestions) for `owl_footage`. Suggestions are for Alex; nothing is changed."""
    if not rows:
        return ["OWLv2 on your footage: no gold-set answers found yet"], []
    lines = ["OWLv2 on your footage (from your Yes/No answers in the gold-set notebook; photos it got "
             "fully right = every box right, none missing):"]
    sugg = []
    for folder, r in sorted(rows.items()):
        n = r["yes"] + r["no"]
        line = f"  {folder}: {_share(r['yes'], n)} photos fully right"
        if r["wrong"] or r["missing"]:
            line += f"; {r['wrong']} had a wrong box, {r['missing']} had an object with no box"
        if r["rejected"] is not None:
            line += f"; {r['rejected']} footage frames where OWLv2 found nothing"
        lines.append(line)
        if n >= OWL_SUGGEST_MIN and r["yes"] / n < OWL_SUGGEST:
            obj = folder if folder in OBJECT_NAMES else None
            if obj and r["missing"] > r["wrong"]:
                sugg.append(f"suggestion: OWLv2 often leaves {obj} without a box; a lower OWLv2 cut-off "
                            f"for {obj} (autobox.THRESHOLD, now {cutoff(threshold, obj)}) may find more. "
                            "Your decision; nothing was changed")
            else:
                sugg.append(f"suggestion: OWLv2 got only {_pct(r['yes'] / n)} of your {folder} photos fully "
                            "right; look at that folder's contact sheet before trusting its boxes. "
                            "Nothing was changed")
    return lines, sugg


def gold_id(manifest):
    """Short id of a gold set: changes if any gold file changes."""
    h = hashlib.sha256("\n".join(sorted(e["sha256"] for e in manifest["images"])).encode())
    return h.hexdigest()[:12]


def file_id(path):
    from .gold import file_hash
    return file_hash(path)[:12]


def run_labeller_check(models_dir, manifest, gold_samps, detector_id, make_labellers,
                       own_root=None, extra_id=None, classes=OBJECT_NAMES, log=print):
    """Labeller check with a cache (Drive `labeller_check.json`): the labellers run only when
    the gold set, the detector version (or `extra_id`, e.g. OWLv2 queries) changed.
    make_labellers() -> (det, owl, det_cut). The OWLv2-on-footage block is rebuilt every time
    from `gold_answers.json` (cheap). Saves and returns the report lines."""
    models_dir = Path(models_dir)
    path = models_dir / LABELLER_FILE
    old = _read(path, {})
    key = {"gold": gold_id(manifest), "detector": detector_id, "extra": extra_id}
    if old.get("key") == key and old.get("result"):
        res = old["result"]
        log("labeller check: gold set and detector unchanged, reusing the saved numbers")
    else:
        det, owl, det_cut = make_labellers()
        res = labeller_check(gold_samps, det, owl, det_cut, classes=classes)
    lines = labeller_lines(res, classes)
    o_lines, sugg = owl_footage_lines(owl_footage(_read(models_dir / GOLD_ANSWERS, {}), manifest, own_root, classes))
    lines += o_lines + sugg
    _write(path, {"key": key, "date": date.today().isoformat(), "result": res, "lines": lines})
    return lines


def saved_labeller_lines(models_dir):
    return _read(Path(models_dir) / LABELLER_FILE, {}).get("lines") or []


# ------------------------------------------------------------------ the report

def report(walks_root, models_dir, classes=OBJECT_NAMES):
    """Plain-language drift report (list of lines, warnings first). Also rewrites `audit.json`
    and saves the report as `drift_report.txt`."""
    models_dir = Path(models_dir)
    rows = answered(walks_root)
    write_audit_log(walks_root, models_dir)
    a_lines, a_warns = audit_lines([r for r in rows if r["audit"]], classes)
    g_lines, g_warns = gold_lines(load_history(models_dir))
    counts = label_counts(walks_root, classes)
    m_warns = missing_warnings(counts, classes)
    ag_lines, ag_warns = [], []
    for c, series in agreement_series(walks_root, classes).items():
        note, warn = agreement_warning(c, series)
        ag_lines.append(note)
        if warn:
            ag_warns.append(warn)

    warns = g_warns + a_warns + ag_warns + m_warns
    lines = ["Drift report (watches for the loop teaching itself mistakes; never blocks anything)"]
    lines += warns or ["no warnings"]
    lines += g_lines
    lines.append("spot checks of accepted auto-labels:")
    lines += a_lines
    lines.append(review_no_line(rows))
    lines.append("detector-OWLv2 agreement rate per object:")
    lines += ag_lines
    if counts:
        lines.append("self-labelled boxes in trainable photos, last walks:")
        for w, c in counts[-5:]:
            lines.append(f"  {w}: " + ", ".join(f"{k} {c.get(k, 0)}" for k in classes))
    lines += saved_labeller_lines(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / REPORT_FILE).write_text("\n".join(lines) + "\n")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Drift report for the learning loop")
    ap.add_argument("--walks", required=True, help="walk_frames folder (one sub-folder per walk)")
    ap.add_argument("--models-dir", required=True)
    a = ap.parse_args(argv)
    print("\n".join(report(a.walks, a.models_dir)))


if __name__ == "__main__":
    main()
