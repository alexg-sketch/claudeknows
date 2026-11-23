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

    python -m claudeknows.drift --walks DRIVE/claudeknows_models/walk_frames --models-dir DRIVE/claudeknows_models
"""
import argparse
import json
import math
from datetime import date
from pathlib import Path

from .objects import OBJECT_NAMES
from .selflabel import load_labels, load_stats, walk_dirs

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
