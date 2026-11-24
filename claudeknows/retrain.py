"""Retrain + quality gate (roadmap task 27, Phase 2).

One loop run trains ONE candidate detector on the approved data (the original
human/Open Images training set as replay, plus at most 30% self-labelled walk
frames from task 26), then compares it with the current model on the frozen gold
set (task 25). The candidate replaces the current model only if it is clearly
better:

  - the overall gold-set difference (paired, resampling whole videos/photos,
    5000 times) has a 95% range whose LOW end is above 0;
  - no object's low end is below -2 points;
  - the overall gain beats the "noise floor" (2.5 x the standard deviation of gold
    mAP over 5 seeds of the same recipe, measured once with `noise-floor`; each seed
    is saved as soon as it is done, so a disconnect costs one training at most);
  - the gold set shows at least 5 different scenes (else: "not enough evidence").

Everything is kept in one folder (Drive `claudeknows_models/`):
  detector_YYYY-MM-DD.pt   every candidate, dated (never deleted)
  detector.pt              copy of the current (promoted) model, what the notebooks load
  versions.json            which versions exist, which is current, the promotion order
  gold_log.json            one row per look at the gold set (gate checks and noise-floor runs)
  noise_floor.json         the noise floor and the recipe it was measured with

The gold set is only used for the yes/no promotion. Seeds, `--init`, augmentation,
score cut-offs and early stopping are all chosen on the validation split, never on
gold, and only one candidate meets the gold set per run (RESEARCH.md 2026-11-13).

    python -m claudeknows.retrain run --root data/detection --walks walks --models-dir DRIVE/claudeknows_models
    python -m claudeknows.retrain noise-floor --root data/detection --models-dir DRIVE/claudeknows_models
    python -m claudeknows.retrain rollback --models-dir DRIVE/claudeknows_models
    python -m claudeknows.retrain status --models-dir DRIVE/claudeknows_models
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from .objects import OBJECT_NAMES

GATE_RESAMPLES = 5000
MIN_GOLD_SCENES = 5
OBJECT_MARGIN = 0.02  # an object may not get worse by more than 2 points (low end of its range)
GOLD_LOOK_BUDGET = 20  # after this many gate checks, warn that gold scores are getting optimistic
REPLAY_MIN_SHARE = 0.5  # the original human/Open Images photos are at least half of training
NOISE_SEEDS = (0, 1, 2, 3, 4)
NOISE_SD_FACTOR = 2.5  # floor = 2.5 x standard deviation of the seeds' gold mAP
INITS = ("coco", "current")
CURRENT = "detector.pt"
VERSIONS_FILE = "versions.json"
GOLD_LOG = "gold_log.json"
NOISE_FILE = "noise_floor.json"
ORIGINAL = "detector_original.pt"  # the detector.pt that existed before versioning started


class OneCandidateOnly(RuntimeError):
    """Raised when a second candidate tries to meet the gold set in the same run."""


def _read(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _pts(x):
    return f"{100 * x:+.1f}"


# ------------------------------------------------------------------ training data

def mix_training(base, extra, max_share=None, seed=0, log=print):
    """Original training photos (replay) + a capped random pick of self-labelled frames.

    Self-labelled frames are at most `max_share` (30%) of the result, so the original
    set is always at least REPLAY_MIN_SHARE (half) of it."""
    from .selflabel import MAX_SELF_SHARE, cap_self_share
    max_share = MAX_SELF_SHARE if max_share is None else max_share
    mixed = cap_self_share(base, extra, max_share, seed, log)
    if mixed and len(base) / len(mixed) < REPLAY_MIN_SHARE:
        raise ValueError(f"original photos are only {len(base)} of {len(mixed)}: "
                         f"need at least {int(REPLAY_MIN_SHARE * 100)}%")
    return mixed


def start_weights(init, current=None):
    """Weights a candidate starts from: None = fresh (COCO weights + new head, the default),
    or the current model's state dict for `init="current"`."""
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}")
    if init == "coco":
        return None
    if current is None or not Path(current).exists():
        raise ValueError("init='current' needs a current model, and there is none yet")
    from .detector import load_detector
    return load_detector(current)[0].state_dict()


# ------------------------------------------------------------------ the gate

def gold_scene_count(gold_samps):
    """Number of different places in the gold set (videos of the same room count once)."""
    from .detection_data import scene_groups
    gold_samps = list(gold_samps)
    if not gold_samps:
        return 0
    scenes = scene_groups(gold_samps)
    return len({scenes.get(s.group, s.group) for s in gold_samps})


def decide(diffs, n_scenes, floor, classes=OBJECT_NAMES, min_scenes=MIN_GOLD_SCENES,
           margin=OBJECT_MARGIN, floor_note=None):
    """Promotion rule. `diffs` from `detector.paired_differences` (candidate - current).

    Returns {"promote": bool, "reasons": [plain-language reasons it was NOT promoted],
    "overall": (difference, low, high)}."""
    reasons = []
    overall = diffs.get("overall")
    if n_scenes < min_scenes:
        reasons.append(f"not enough evidence: the gold set shows only {n_scenes} scene(s), "
                       f"{min_scenes} needed")
    if overall is None:
        reasons.append("no gold-set score could be computed")
    else:
        d, lo, hi = overall
        if lo <= 0:
            reasons.append(f"not clearly better overall ({_pts(d)} points, range "
                           f"{_pts(lo)} to {_pts(hi)})")
        if floor is None:
            reasons.append(floor_note or "no noise floor measured yet (run `retrain noise-floor` once)")
        elif d <= floor:
            reasons.append(f"gain {_pts(d)} points is within the noise between seeds "
                           f"({100 * floor:.1f} points)")
    for c in classes:
        r = diffs.get(c)
        if r is not None and r[1] < -margin:
            reasons.append(f"{c} may have got worse ({_pts(r[0])} points, range "
                           f"{_pts(r[1])} to {_pts(r[2])})")
    return {"promote": not reasons, "reasons": reasons, "overall": overall}


def gold_map(results, classes=OBJECT_NAMES):
    """Mean of the per-object AP@0.5 (objects without boxes left out), or None."""
    from .detector import average_precision
    aps = [a for a in (average_precision(results, c) for c in classes) if a is not None]
    return round(sum(aps) / len(aps), 4) if aps else None


def promotion_message(decision):
    """Honest headline: a promotion's gain is the LOW end of the range ("at least")."""
    o = decision["overall"]
    if decision["promote"]:
        return (f"promoted: at least {_pts(o[1])} mAP points better on the gold set "
                f"(middle estimate {_pts(o[0])})")
    return "kept the current model: " + "; ".join(decision["reasons"])


def load_gold_log(models_dir):
    return _read(Path(models_dir) / GOLD_LOG, [])


def log_gold_check(models_dir, row):
    rows = load_gold_log(models_dir)
    rows.append({"date": date.today().isoformat(), **row})
    _write(Path(models_dir) / GOLD_LOG, rows)
    return rows


def gold_use_lines(models_dir, created=None, budget=GOLD_LOOK_BUDGET):
    """Summary lines: how often the gold set was used, plus a wear warning after `budget` gates."""
    rows = load_gold_log(models_dir)
    gates = sum(1 for r in rows if r.get("kind") == "gate")
    since = f" since {created}" if created else ""
    lines = [f"gold set used {len(rows)} times{since} ({gates} promotion checks, "
             f"{len(rows) - gates} noise-floor runs)"]
    if gates >= budget:
        lines.append(f"WARNING: the gold set has been used for {budget}+ decisions; its scores are "
                     "getting optimistic. Consider adding fresh checked photos "
                     "(see RESEARCH.md 2026-11-13)")
    return lines


class GoldGate:
    """The single gold-set check of one loop run. A second candidate is refused."""

    def __init__(self, models_dir, classes=OBJECT_NAMES, n=GATE_RESAMPLES, seed=0):
        self.models_dir, self.classes, self.n, self.seed = Path(models_dir), classes, n, seed
        self.used_by = None

    def check(self, name, current_results, candidate_results, n_scenes, floor, floor_note=None):
        from .detector import paired_differences
        if self.used_by is not None:
            raise OneCandidateOnly(f"the gold set was already used this run (for {self.used_by}); "
                                   f"only one candidate per run may be compared on it")
        self.used_by = name
        diffs = paired_differences(current_results, candidate_results, self.classes, self.n, self.seed)
        decision = decide(diffs, n_scenes, floor, self.classes, floor_note=floor_note)
        decision["diffs"] = diffs
        o = decision["overall"]
        log_gold_check(self.models_dir, {
            "kind": "gate", "candidate": name,
            "verdict": "promoted" if decision["promote"] else "kept current",
            "difference": None if o is None else round(o[0], 4),
            "range": None if o is None else [round(o[1], 4), round(o[2], 4)]})
        return decision


# ------------------------------------------------------------------ noise floor

def noise_floor_value(gold_maps, factor=NOISE_SD_FACTOR):
    """2.5 x the sample standard deviation of the seeds' gold mAP (RESEARCH.md 2026-11-14).
    Unlike max - min it does not grow when seeds are added."""
    import numpy as np
    return float(factor * np.std(np.asarray(gold_maps, dtype=float), ddof=1))


def _recipe_core(recipe):
    return {k: v for k, v in (recipe or {}).items() if k != "seeds"}


def load_noise_data(models_dir):
    """The saved per-seed measurements as {"recipe", "seeds": {seed: gold mAP}, "floor", "date"}.
    A file from the old 3-seed rule (a list of gold mAPs) is read as seeds 0-2, so it is
    topped up with the missing seeds rather than measured again. None if nothing saved."""
    data = _read(Path(models_dir) / NOISE_FILE, None)
    if data is None:
        return None
    if isinstance(data.get("gold_maps"), list):  # old format
        seeds = data.get("recipe", {}).get("seeds", list(range(len(data["gold_maps"]))))
        data = {"recipe": _recipe_core(data.get("recipe")), "date": data.get("date"),
                "seeds": {str(k): float(m) for k, m in zip(seeds, data["gold_maps"])}}
    data["recipe"] = _recipe_core(data.get("recipe"))
    data.setdefault("seeds", {})
    data["floor"] = (noise_floor_value(list(data["seeds"].values()))
                     if all(str(k) in data["seeds"] for k in NOISE_SEEDS) else None)
    return data


def noise_progress(models_dir, recipe=None, seeds=NOISE_SEEDS, log=print):
    """(floor or None, measured seeds, missing seeds, plain-language note).

    Measurements made with different training settings do not count (the floor must be
    measured again for the new recipe)."""
    data = load_noise_data(models_dir)
    if data is not None and recipe is not None and data["recipe"] != _recipe_core(recipe):
        log("the noise floor was measured with different training settings: measure it again")
        data = None
    done = sorted(int(k) for k in (data or {}).get("seeds", {}) if int(k) in seeds)
    missing = [s for s in seeds if s not in done]
    floor = data["floor"] if data is not None and not missing else None
    if floor is not None:
        note = f"noise floor: {100 * floor:.1f} points ({len(done)} seeds)"
    elif not done:
        note = f"no noise floor measured yet (run `retrain noise-floor` once; {len(seeds)} trainings)"
    else:
        note = (f"noise floor: {len(done)} of {len(seeds)} seeds measured (the next "
                f"`retrain noise-floor` trains only the missing ones)")
    return floor, done, missing, note


def record_noise_seed(models_dir, seed, gold_map, recipe):
    """Save ONE seed's gold mAP straight away (atomic write) and log it as one gold use, so a
    Colab disconnect loses at most the training that was running. Returns the saved data."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    core = _recipe_core(recipe)
    data = load_noise_data(d)
    if data is None or data["recipe"] != core:
        data = {"recipe": core, "seeds": {}}
    data["seeds"][str(seed)] = float(gold_map)
    data["date"] = date.today().isoformat()
    log_gold_check(d, {"kind": "noise-floor", "candidate": f"seed {seed}", "verdict": None,
                       "difference": None, "range": None, "gold_map": round(float(gold_map), 4)})
    done = all(str(k) in data["seeds"] for k in NOISE_SEEDS)
    data["floor"] = noise_floor_value(list(data["seeds"].values())) if done else None
    _write(d / NOISE_FILE, data)
    return data


def record_noise_floor(models_dir, gold_maps, recipe):
    """Save several seeds at once (seeds from `recipe["seeds"]`, default 0, 1, ...)."""
    seeds = recipe.get("seeds", list(range(len(gold_maps))))
    data = None
    for s, m in zip(seeds, gold_maps):
        data = record_noise_seed(models_dir, s, m, recipe)
    return data


def load_noise_floor(models_dir, recipe=None, log=print):
    """The floor, or None if not (fully) measured or measured with a different recipe."""
    return noise_progress(models_dir, recipe, log=log)[0]


# ------------------------------------------------------------------ versions

def load_versions(models_dir):
    return _read(Path(models_dir) / VERSIONS_FILE, {"current": None, "promoted": [], "all": []})


def current_path(models_dir):
    p = Path(models_dir) / CURRENT
    return p if p.exists() else None


def save_version(model, models_dir, history=(), extra=None, today=None):
    """Save a candidate as detector_YYYY-MM-DD.pt (then _2, _3 ... the same day). Returns its name."""
    from .detector import save_detector
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"detector_{(today or date.today()).isoformat()}"
    name, i = f"{stem}.pt", 2
    while (d / name).exists():
        name, i = f"{stem}_{i}.pt", i + 1
    save_detector(model, d / name, history, extra=extra)
    v = load_versions(d)
    entry = {"name": name, "date": (today or date.today()).isoformat(),
             "init": (extra or {}).get("init"), "verdict": None}
    if (extra or {}).get("training_set"):
        entry["training_set"] = extra["training_set"]  # which saved original photos (task 29d)
    v["all"].append(entry)
    _write(d / VERSIONS_FILE, v)
    return name


def _set_verdict(models_dir, name, verdict):
    v = load_versions(models_dir)
    for e in v["all"]:
        if e["name"] == name:
            e["verdict"] = verdict
    _write(Path(models_dir) / VERSIONS_FILE, v)


def promote(models_dir, name, log=print):
    """Make `name` the current model (copied to detector.pt). A detector.pt from before
    versioning is first kept as detector_original.pt so it can be rolled back to."""
    d = Path(models_dir)
    v = load_versions(d)
    if (d / CURRENT).exists() and not v["promoted"]:
        shutil.copy2(d / CURRENT, d / ORIGINAL)
        v["promoted"].append(ORIGINAL)
        v["all"].insert(0, {"name": ORIGINAL, "date": None, "init": None, "verdict": "original"})
    shutil.copy2(d / name, d / CURRENT)
    v["promoted"].append(name)
    v["current"] = name
    _write(d / VERSIONS_FILE, v)
    log(f"current model is now {name}")
    return v


def rollback(models_dir, log=print):
    """Go back to the model promoted before the current one. Returns its name (or None)."""
    d = Path(models_dir)
    v = load_versions(d)
    if len(v["promoted"]) < 2:
        log("nothing to roll back to: there is only one promoted version")
        return None
    dropped = v["promoted"].pop()
    back = v["promoted"][-1]
    shutil.copy2(d / back, d / CURRENT)
    v["current"] = back
    for e in v["all"]:
        if e["name"] == dropped:
            e["verdict"] = "rolled back"
    _write(d / VERSIONS_FILE, v)
    log(f"rolled back from {dropped} to {back}")
    return back


# ------------------------------------------------------------------ one loop run

def recipe_of(train_kwargs, pretrained=True, training_set=None):
    """The settings that define "the recipe" for the noise floor (seed excluded).
    `training_set` = id of the saved original training photos (task 29d): new photos are a
    new recipe, so the noise floor is measured again."""
    keys = ("epochs", "batch_size", "lr", "composite", "rot_degrees", "rot90_p", "neg_frac",
            "zoom_crop_p", "ema_decay", "patience", "max_batches")
    recipe = {"pretrained": pretrained, **{k: train_kwargs.get(k) for k in keys}}
    if training_set:
        recipe["training_set"] = training_set
    return recipe


def _train(base, extra, seed, init_state, pretrained, gold_manifest, train_kwargs, val_frac, log):
    from .detection_data import scene_groups, split_detection
    from .detector import best_thresholds, train_detector
    from .selflabel import self_box_usage
    fit, val = (split_detection(base, val_frac, seed, scenes=scene_groups(base), log=log)
                if val_frac > 0 else (list(base), []))
    n_orig = len(fit)
    fit = mix_training(fit, extra, seed=seed, log=log)  # validation stays original photos only
    stats = {}
    model, hist = train_detector(fit, val_samples=val or None, seed=seed, pretrained=pretrained,
                                 init_state=init_state, gold=gold_manifest, stats=stats, log=log,
                                 **train_kwargs)
    cuts = best_thresholds(model, val) if val else {}  # cut-offs from validation, never gold
    model.thresholds = cuts
    stats["self_box_usage"] = self_box_usage(extra, fit[n_orig:]) if extra else None
    return model, hist, stats, cuts, len(fit), len(fit) - n_orig


def retrain(base, extra, gold_samps, models_dir, *, gold_manifest=None, init="coco", seed=0,
            pretrained=True, val_frac=0.15, train_kwargs=None, gate=None, tta=None, today=None,
            training_set=None, log=print):
    """Train one candidate, compare it on the gold set, promote it or not.

    `base` = original training photos (gold already removed), `extra` = self-labelled walk
    frames (`selflabel.self_samples`), `gold_samps` = `gold.gold_samples(manifest)`.
    Returns {"name", "promoted", "decision", "summary" (list of plain-language lines)}.
    """
    from .detector import load_detector, predict_samples
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    train_kwargs = dict(train_kwargs or {})
    current = current_path(d)
    model, hist, stats, cuts, n_fit, n_self = _train(base, extra, seed, start_weights(init, current),
                                             pretrained, gold_manifest, train_kwargs, val_frac, log)
    name = save_version(model, d, hist, today=today, extra={
        "train_stats": stats, "thresholds": cuts, "init": init, "seed": seed,
        "recipe": recipe_of(train_kwargs, pretrained, training_set), "self_labelled": n_self,
        "trained_on": n_fit, "training_set": training_set})
    summary = [f"candidate {name}: trained on {n_fit} photos ({n_self} self-labelled), "
               f"started from {'COCO weights (fresh)' if init == 'coco' else 'the current model'}"]
    if stats.get("self_box_usage"):
        summary.append(stats["self_box_usage"])
    created = (gold_manifest or {}).get("created")
    result = {"name": name, "promoted": False, "decision": None, "summary": summary,
              "gold": None, "gold_created": created}
    if current is None:
        promote(d, name, log)
        _set_verdict(d, name, "first version")
        summary.append("no current model yet: this candidate is now the current model "
                       "(the gold-set check starts with the next one)")
        result["promoted"] = True
        return result
    gold_samps = list(gold_samps or [])
    if not gold_samps:
        _set_verdict(d, name, "kept current")
        summary.append("kept the current model: there is no gold set yet, so nothing can be "
                       "promoted automatically")
        return result
    floor, _, _, floor_note = noise_progress(d, recipe_of(train_kwargs, pretrained, training_set),
                                             log=log)
    cur_model, classes = load_detector(current)
    cur_res = predict_samples(cur_model, gold_samps, tta=tta)
    cand_res = predict_samples(model, gold_samps, tta=tta)
    gate = gate or GoldGate(d, classes)
    decision = gate.check(name, cur_res, cand_res, gold_scene_count(gold_samps), floor, floor_note)
    result["decision"] = decision
    result["gold"] = {"current": gold_map(cur_res, classes), "candidate": gold_map(cand_res, classes),
                      "current_name": load_versions(d).get("current")}
    if decision["promote"]:
        promote(d, name, log)
        result["promoted"] = True
    _set_verdict(d, name, "promoted" if decision["promote"] else "kept current")
    summary.append(promotion_message(decision))
    for c in classes:
        r = decision["diffs"].get(c)
        if r is not None:
            summary.append(f"  {c}: {_pts(r[0])} points (range {_pts(r[1])} to {_pts(r[2])})")
    summary += gold_use_lines(d, created)
    return result


def measure_noise_floor(base, gold_samps, models_dir, *, gold_manifest=None, seeds=NOISE_SEEDS,
                        pretrained=True, val_frac=0.15, train_kwargs=None, train_fn=None,
                        training_set=None, log=print):
    """Train the current recipe (fresh start, original photos only) with each seed that is
    still MISSING, score it on the gold set once and save it at once. When all seeds exist
    the floor is 2.5 x their standard deviation. These models are never promotion
    candidates and are not saved. `train_fn(seed)` -> model replaces the training (tests)."""
    from .detector import average_precision, predict_samples
    classes = OBJECT_NAMES
    train_kwargs = dict(train_kwargs or {})
    recipe = {**recipe_of(train_kwargs, pretrained, training_set), "seeds": list(seeds)}
    _, done, missing, _ = noise_progress(models_dir, recipe, seeds, log)
    if done:
        log(f"noise floor: seeds {done} already measured; training {missing or 'none'}")
    for seed in missing:
        if train_fn is not None:
            model = train_fn(seed)
        else:
            model = _train(base, [], seed, None, pretrained, gold_manifest, train_kwargs, val_frac, log)[0]
        res = predict_samples(model, list(gold_samps))
        aps = [a for a in (average_precision(res, c) for c in classes) if a is not None]
        m = sum(aps) / len(aps) if aps else 0.0
        record_noise_seed(models_dir, seed, m, recipe)
        log(f"seed {seed}: gold mAP {100 * m:.1f}% (saved)")
    floor, _, _, note = noise_progress(models_dir, recipe, seeds, log)
    log(note)
    return load_noise_data(models_dir)


# ------------------------------------------------------------------ command line

def _train_args(p):
    p.add_argument("--root", default="data/detection")
    p.add_argument("--gold-dir", default=None, help="where the frozen gold set lives; default --root")
    p.add_argument("--models-dir", required=True, help="Drive claudeknows_models/")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")


def _data(a, log=print):
    from .detection_data import scan_detection, scan_negatives, scene_groups, split_detection
    from .gold import gold_samples, load_gold, remove_gold, verify_gold
    gold_dir = a.gold_dir or a.root
    gold = load_gold(gold_dir)
    samples, _ = scan_detection(a.root)
    if gold:
        samples = remove_gold(samples, gold, log)
        for prob in verify_gold(gold, gold_dir):
            log(f"WARNING gold set {prob}")
    # same split as `detector train`, so the normal test set stays out of training
    base, _ = split_detection(samples, scenes=scene_groups(samples), min_own_test=1, log=log)
    negs, _ = split_detection(scan_negatives(a.root), log=log)
    gsamps = gold_samples(gold, gold_dir) if gold else []
    kwargs = dict(epochs=a.epochs, batch_size=a.batch_size, patience=a.patience,
                  workers=a.workers, max_batches=a.max_batches, negatives=negs or None)
    return base, gsamps, gold, kwargs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train one candidate and promote it only if it passes the gate")
    _train_args(r)
    r.add_argument("--walks", default=None, help="folder of auto-labelled walks (task 26)")
    r.add_argument("--init", choices=INITS, default="coco")
    r.add_argument("--seed", type=int, default=0)
    n = sub.add_parser("noise-floor", help="measure once: gold mAP spread over 5 seeds (trains only missing seeds)")
    _train_args(n)
    for name in ("rollback", "status"):
        s = sub.add_parser(name)
        s.add_argument("--models-dir", required=True)
    a = p.parse_args(argv)
    if a.cmd == "rollback":
        rollback(a.models_dir)
        return
    if a.cmd == "status":
        v = load_versions(a.models_dir)
        print(f"current model: {v['current'] or 'none'}")
        for e in v["all"]:
            print(f"  {e['name']}  {e.get('verdict') or ''}  (start: {e.get('init') or '?'})")
        print("\n".join(gold_use_lines(a.models_dir)))
        return
    base, gsamps, gold, kwargs = _data(a)
    if a.cmd == "noise-floor":
        measure_noise_floor(base, gsamps, a.models_dir, gold_manifest=gold,
                            pretrained=not a.no_pretrained, train_kwargs=kwargs)
        return
    from .selflabel import self_samples
    extra = self_samples(a.walks, gold) if a.walks else []
    out = retrain(base, extra, gsamps, a.models_dir, gold_manifest=gold, init=a.init, seed=a.seed,
                  pretrained=not a.no_pretrained, train_kwargs=kwargs)
    print("\n".join(out["summary"]))


if __name__ == "__main__":
    main()
