"""Training data for the new objects: internet photos + Alex's own footage.

Layout (everything under `data/`, which git ignores; the repo is public, so
personal photos/videos must never be committed):

    data/objects/web/<object>/<ImageID>.jpg      from `claudeknows.openimages`
    data/objects/own/<object>/photo_<name>.jpg  Alex's photos
    data/objects/own/<object>/vid_<name>__f0001.jpg   frames of Alex's videos

`import_own(drive_dir, root)` fills `own/` from a folder such as Google Drive's
`claudeknows_photos/` (sub-folders `water_bottle/`, `keys/`, `shoes/`). A
missing or empty folder is fine: that object just trains on internet photos.

Splitting is done by *group*, never by frame: all frames of one video, or one
photo, stay together, either all in train or all in test.
"""
import argparse
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .frames import MAX_SIDE, PHOTO_EXTS, VIDEO_EXTS, extract_frames

OBJECT_NAMES = ("water_bottle", "keys", "shoes")
SOURCES = ("web", "own")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str  # "web" or "own"
    group: str  # image id, photo name, or video name


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "x"


def _open_photo(path):
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        if Path(path).suffix.lower() == ".heic":
            try:  # iPhone photos; needs `pip install pillow-heif`
                from pillow_heif import register_heif_opener
                register_heif_opener()
                img = Image.open(path)
                img.load()
            except Exception:
                return None
        else:
            return None
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def import_own(drive_dir, root="data/objects", objects=OBJECT_NAMES, fps=2.0,
               backend="auto", log=print):
    """Copy Alex's photos and turn his videos into frames. Returns a summary dict.

    Safe to re-run (done videos/photos are skipped). Missing folders are ignored.
    """
    drive_dir, root = Path(drive_dir), Path(root)
    summary = {}
    for obj in objects:
        src = drive_dir / obj
        out = root / "own" / obj
        info = summary[obj] = {"photos": 0, "videos": 0, "frames": 0,
                               "blurry_skipped": 0, "unreadable": []}
        if not src.is_dir():
            log(f"{obj}: no folder at {src} (using internet photos only)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            ext = f.suffix.lower()
            if f.name.startswith(".") or not f.is_file():
                continue
            if ext in PHOTO_EXTS:
                dest = out / f"photo_{_safe(f.stem)}.jpg"
                if not dest.exists():
                    img = _open_photo(f)
                    if img is None:
                        info["unreadable"].append(f.name)
                        continue
                    img.save(dest, quality=92)
                info["photos"] += 1
            elif ext in VIDEO_EXTS:
                prefix = f"vid_{_safe(f.stem)}"
                if list(out.glob(prefix + "__f*.jpg")):
                    info["videos"] += 1
                    continue
                try:
                    kept, skipped = extract_frames(f, out, fps=fps, prefix=prefix,
                                                   backend=backend)
                except Exception:
                    info["unreadable"].append(f.name)
                    continue
                if kept == 0:
                    info["unreadable"].append(f.name)
                    continue
                info["videos"] += 1
                info["frames"] += kept
                info["blurry_skipped"] += skipped
        log(f"{obj}: {info['photos']} photos, {info['videos']} videos "
            f"({info['frames']} new frames, {info['blurry_skipped']} blurry skipped)"
            + (f", could not read: {info['unreadable']}" if info["unreadable"] else ""))
    return summary


def _group_of(path):
    stem = Path(path).stem
    return stem.split("__f")[0] if stem.startswith("vid_") else stem


def scan(root="data/objects", classes=OBJECT_NAMES):
    """List every sample under root/{web,own}/<class>/. Returns (classes, samples)."""
    root = Path(root)
    samples = []
    for source in SOURCES:
        for label, cls in enumerate(classes):
            for f in sorted((root / source / cls).glob("*.jpg")):
                samples.append(Sample(f, label, source, _group_of(f)))
    return tuple(classes), samples


def split_by_group(samples, test_frac=0.2, seed=0):
    """Split into (train, test) so no group (video or photo) is in both.

    Done per (class, source), so each source is represented in both sides where
    it has two or more groups. An object with a single video of Alex's keeps it
    all in train (there is nothing to hold out) - the evaluation then simply has
    no "own" test photos for it.
    """
    buckets = {}
    for s in samples:
        buckets.setdefault((s.label, s.source), {}).setdefault(s.group, []).append(s)
    train, test = [], []
    for key in sorted(buckets):
        groups = sorted(buckets[key])
        # stable shuffle that does not depend on Python's hash seed
        groups.sort(key=lambda g: hashlib.md5(f"{seed}:{key}:{g}".encode()).hexdigest())
        n_test = round(len(groups) * test_frac)
        if len(groups) >= 2:
            n_test = max(1, n_test)
        else:
            n_test = 0
        for i, g in enumerate(groups):
            (test if i < n_test else train).extend(buckets[key][g])
    random.Random(seed).shuffle(train)
    return train, test


class ObjectDataset(Dataset):
    """Images + class index; also exposes `.samples` for per-source reporting."""

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, s.label


def describe(classes, train, test):
    """Plain-text table of how many photos each object/source has in each split."""
    lines = [f"{'object':<14}{'source':<6}{'train':>7}{'test':>6}{'groups':>8}"]
    for label, cls in enumerate(classes):
        for source in SOURCES:
            tr = [s for s in train if s.label == label and s.source == source]
            te = [s for s in test if s.label == label and s.source == source]
            groups = len({s.group for s in tr + te})
            lines.append(f"{cls:<14}{source:<6}{len(tr):>7}{len(te):>6}{groups:>8}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Import Alex's own photos/videos (e.g. from Google Drive).")
    p.add_argument("drive_dir", help="folder with water_bottle/, keys/, shoes/ sub-folders")
    p.add_argument("--root", default="data/objects")
    p.add_argument("--fps", type=float, default=2.0)
    a = p.parse_args(argv)
    import_own(a.drive_dir, a.root, fps=a.fps)


if __name__ == "__main__":
    main()
