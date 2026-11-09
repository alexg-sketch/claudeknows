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
import shutil
from dataclasses import replace
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


CHECKED = "checked"


def candidates(samples, max_per_folder=60):
    """Alex's own footage that could become gold (still needs his yes: see `mark_checked`)."""
    picked, counts = [], {}
    for s in samples:
        if s.source == "own" and s.by != "pending" and counts.get(s.folder, 0) < max_per_folder:
            picked.append(s)
            counts[s.folder] = counts.get(s.folder, 0) + 1
    return picked


def mark_checked(samples, confirmed):
    """Samples Alex confirmed (`confirmed` = file names or paths) with their boxes
    stamped by="checked". Anything not confirmed is not returned, so never gold."""
    keep = {str(c) for c in confirmed}
    return [replace(s, by=CHECKED) for s in samples if s.path.name in keep or str(s.path) in keep]


def review_sheet(samples, out="gold_review.jpg", cols=5, thumb=240):
    """One picture of the candidate photos with their boxes drawn and their file names
    printed, so Alex can say which ones are right."""
    from PIL import Image, ImageDraw
    rows = max(1, -(-len(samples) // cols))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, s in enumerate(samples):
        with Image.open(s.path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb - 14))
        d = ImageDraw.Draw(im)
        w, h = im.size
        for name, (x0, y0, x1, y1) in s.boxes:
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline="red", width=2)
            d.text((x0 * w + 2, y0 * h + 2), name, fill="red")
        cell = Image.new("RGB", (thumb, thumb), "white")
        cell.paste(im, (0, 0))
        ImageDraw.Draw(cell).text((2, thumb - 12), s.path.name[:36], fill="black")
        sheet.paste(cell, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out)
    return out


def copy_gold(manifest, root, dest, log=print):
    """Keep the gold set outside Colab's temporary disk: copy every gold image and
    `gold.json` into `dest` (e.g. Drive `claudeknows_models/gold/`). Never overwrites
    files already there, so the saved copy stays the one fixed yardstick."""
    dest = Path(dest)
    n = 0
    for e in manifest["images"] + manifest.get("distance", []):
        t = dest / e["path"]
        if t.exists():
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(root) / e["path"], t)
        n += 1
    g = gold_path(dest)
    if not g.exists():
        g.write_text(json.dumps(manifest, indent=1))
    log(f"gold set saved to {dest} ({n} new files)")
    return dest


def build_gold(samples, root="data/detection", distance=True, force=False, log=print,
               require_checked=True):
    """Freeze `samples` (DetSamples under `root`, boxes confirmed by a person, by="checked")
    plus the `distance/` photos.

    Refuses to overwrite an existing gold set unless `force` (a gold set that changes
    is not a fixed yardstick), and refuses machine-drawn boxes unless `require_checked=False`.
    Returns the manifest dict.
    """
    unchecked = [s for s in samples if s.by != CHECKED]
    if unchecked and require_checked:
        raise ValueError(f"{len(unchecked)} image(s) were not checked by a person "
                         f"(first: {unchecked[0].path.name}); confirm them first")
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


def gold_groups(manifest):
    """(source, group) of every gold image. For Alex's footage a group is one video or
    photo: frames of the same video are near-copies, so a whole gold video stays out
    of training, not just the frames that were frozen."""
    return {(e["source"], e["group"]) for e in manifest["images"]}


def is_gold(sample, manifest):
    """True if the sample is a gold image (same bytes, whatever it is called) or comes
    from the same video/photo as one (frames are re-extracted each Colab session, so
    their bytes may differ, and neighbouring frames of a gold video are near-copies)."""
    if (sample.source, sample.group) in gold_groups(manifest):
        return True
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
    r = sub.add_parser("review", help="make a picture of candidate gold photos for Alex to check")
    r.add_argument("--root", default="data/detection")
    r.add_argument("--max-per-folder", type=int, default=60)
    r.add_argument("--out", default="gold_review.jpg")
    b = sub.add_parser("build", help="freeze the photos Alex confirmed as the gold set")
    b.add_argument("--root", default="data/detection")
    b.add_argument("--max-per-folder", type=int, default=60)
    b.add_argument("--confirm", nargs="+", metavar="FILE", help="file names Alex confirmed")
    b.add_argument("--confirm-file", help="text file with one confirmed file name per line")
    b.add_argument("--save-to", help="also copy the gold set here (Drive)")
    b.add_argument("--force", action="store_true")
    c = sub.add_parser("check", help="verify the frozen files are unchanged")
    c.add_argument("--root", default="data/detection")
    a = p.parse_args(argv)
    if a.cmd == "review":
        samples, _ = scan_detection(a.root)
        cands = candidates(samples, a.max_per_folder)
        print(f"{len(cands)} candidates -> {review_sheet(cands, a.out)}")
    elif a.cmd == "build":
        samples, _ = scan_detection(a.root)
        names = list(a.confirm or [])
        if a.confirm_file:
            names += [l.strip() for l in Path(a.confirm_file).read_text().splitlines() if l.strip()]
        if not names:
            raise SystemExit("nothing confirmed: run `review`, look at the picture, then pass --confirm")
        checked = mark_checked(candidates(samples, 10 ** 9), names)
        m = build_gold(checked, a.root, force=a.force)
        if a.save_to:
            copy_gold(m, a.root, a.save_to)
    else:
        m = load_gold(a.root)
        if m is None:
            print("no gold set yet")
            return
        probs = verify_gold(m, a.root)
        print("gold set OK" if not probs else "\n".join(probs))


if __name__ == "__main__":
    main()
