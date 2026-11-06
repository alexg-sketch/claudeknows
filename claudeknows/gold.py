"""Frozen "gold" test set (roadmap task 25, Phase 2).

The gold set is a fixed list of images Alex has checked. It is written once
(`python -m claudeknows.gold build`), stored with a SHA-256 of every file, and then
 - never trained on (`assert_no_gold` refuses; `detector train` also drops gold images
   from the pool before splitting), and
 - always scored (`detector report` prints the gold scores after the normal test report).

The manifest lives next to the data (default `data/detection/gold.json`), which is
git-ignored: it names Alex's footage, so it must never be committed.
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from .detection_data import DetSample, FOLDERS, load_distance_photos

GOLD_FILE = "gold.json"


class GoldLeak(RuntimeError):
    """Raised when a gold image is about to be used for training."""


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def gold_path(root="data/detection"):
    return Path(root) / GOLD_FILE


def _rel(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def build_gold(samples, root="data/detection", distance=True, force=False, log=print):
    """Freeze `samples` (checked DetSamples under `root`) plus the `distance/` photos.

    Refuses to overwrite an existing gold set unless `force` (a gold set that changes
    is not a fixed yardstick). Returns the manifest dict.
    """
    out = gold_path(root)
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists: the gold set is frozen "
                              "(pass force=True / --force only if you really mean to replace it)")
    items = []
    for s in samples:
        items.append({"path": _rel(s.path, root), "sha256": file_hash(s.path),
                      "label": s.label, "source": s.source, "group": s.group,
                      "folder": s.folder, "by": s.by,
                      "boxes": [[name, list(box)] for name, box in s.boxes]})
    dist = []
    if distance:
        for path, obj, metres in load_distance_photos(root):
            dist.append({"path": _rel(path, root), "sha256": file_hash(path),
                         "object": obj, "metres": metres})
    manifest = {"created": date.today().isoformat(), "images": items, "distance": dist}
    out.write_text(json.dumps(manifest, indent=1))
    log(f"gold set frozen: {len(items)} images, {len(dist)} distance photos -> {out}")
    return manifest


def load_gold(root="data/detection"):
    """The manifest, or None when no gold set exists yet."""
    p = gold_path(root)
    return json.loads(p.read_text()) if p.exists() else None


def gold_hashes(manifest):
    return {e["sha256"] for e in manifest["images"]} | {e["sha256"] for e in manifest.get("distance", [])}


def verify_gold(manifest, root="data/detection"):
    """Plain-language problems: files that went missing or changed since freezing."""
    problems = []
    for e in manifest["images"] + manifest.get("distance", []):
        p = Path(root) / e["path"]
        if not p.exists():
            problems.append(f"missing: {e['path']}")
        elif file_hash(p) != e["sha256"]:
            problems.append(f"changed since freezing: {e['path']}")
    return problems


def gold_samples(manifest, root="data/detection"):
    """The frozen images as DetSamples (for `predict_samples` / `build_report`)."""
    out = []
    for e in manifest["images"]:
        out.append(DetSample(Path(root) / e["path"], e["label"], e["source"], e["group"],
                             e["folder"], tuple((n, tuple(b)) for n, b in e["boxes"]), e["by"]))
    return out


def is_gold(sample, manifest):
    """True if the sample's file is a gold image: same bytes, whatever it is called."""
    return file_hash(sample.path) in gold_hashes(manifest)


def remove_gold(samples, manifest, log=print):
    """Samples without gold images (keeps the gold set out of both train and test pools)."""
    kept = [s for s in samples if not is_gold(s, manifest)]
    if len(kept) != len(samples):
        log(f"left out {len(samples) - len(kept)} gold images")
    return kept


def assert_no_gold(samples, manifest):
    """Refuse to train on any gold image."""
    bad = [str(s.path) for s in samples if is_gold(s, manifest)]
    if bad:
        raise GoldLeak(f"{len(bad)} gold image(s) in the training data, e.g. {bad[0]}")


def main(argv=None):
    from .detection_data import scan_detection
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="freeze Alex's checked own-footage images as the gold set")
    b.add_argument("--root", default="data/detection")
    b.add_argument("--max-per-folder", type=int, default=60)
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("check", help="verify the frozen files are unchanged")
    c.add_argument("--root", default="data/detection")
    a = p.parse_args(argv)
    if a.cmd == "build":
        samples, _ = scan_detection(a.root)
        picked, counts = [], {}
        for s in samples:  # only footage Alex made, with boxes a person has looked at
            if s.source == "own" and s.by != "pending" and counts.get(s.folder, 0) < a.max_per_folder:
                picked.append(s)
                counts[s.folder] = counts.get(s.folder, 0) + 1
        build_gold(picked, a.root, force=a.force)
    else:
        m = load_gold(a.root)
        if m is None:
            print("no gold set yet")
            return
        probs = verify_gold(m, a.root)
        print("gold set OK" if not probs else "\n".join(probs))


if __name__ == "__main__":
    main()
