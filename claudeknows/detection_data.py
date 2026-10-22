"""Training data for finding objects in photos (water bottle / keys / shoes), with boxes.

Unlike `claudeknows.objects` (one cropped object per photo) this keeps WHOLE
photos plus every box in them. Layout (under `data/`, which git ignores; the
repo is public, so personal photos/videos must never be committed):

    data/detection/web/<object>/<ImageID>.jpg   from Open Images, with boxes.json
    data/detection/own/<object>/...              Alex's footage of one object
    data/detection/own/together/...              Alex's videos with 2-3 objects in one shot
    data/detection/own/distance/...              tape-measured photos, TEST ONLY (distances.json)

Every folder with training images has a `boxes.json`:

    {"<file>.jpg": {"boxes": [{"label": "keys", "box": [x0, y0, x1, y1]}, ...],
                    "by": "openimages" | "pending" | "owlv2"}}

Boxes are fractions of the image (0..1), so they survive resizing. `"by":
"pending"` means nobody has boxed the photo yet (Alex's footage, and Open Images
keys, which have no boxes): task 19 fills those in, and until then they are
skipped for training and counted in the report.

    python -m claudeknows.detection_data download --out data/detection --cap 300
    python -m claudeknows.detection_data import /content/drive/MyDrive/claudeknows_photos
"""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from . import openimages
from .frames import PHOTO_EXTS
from .objects import OBJECT_NAMES, _group_of, import_own, split_by_group

DETECT_SIDE = 640  # whole photos are kept bigger than classifier crops: far objects are small
MIN_BOX_SIDE = 0.03  # ignore Open Images boxes thinner than 3% of the photo
TOGETHER = "together"
DISTANCE = "distance"
FOLDERS = OBJECT_NAMES + (TOGETHER,)  # folders that can hold training images
SOURCES = ("web", "own")
BOXES_FILE = "boxes.json"
DISTANCES_FILE = "distances.json"
DIST_NAME = re.compile(
    r"^(?P<obj>water_bottle|keys|shoes)[ _-]+(?P<m>\d+(?:[.,]\d+)?)\s*m(?![a-z])", re.I)


# ---------------------------------------------------------------- boxes.json

def load_boxes(folder):
    f = Path(folder) / BOXES_FILE
    return json.loads(f.read_text()) if f.exists() else {}


def save_boxes(folder, data):
    (Path(folder) / BOXES_FILE).write_text(json.dumps(data, indent=0, sort_keys=True))


def register_pending(folder):
    """Add every image in `folder` that has no entry yet as "pending" (to be auto-boxed).

    Returns the number of newly registered images.
    """
    folder = Path(folder)
    data = load_boxes(folder)
    new = 0
    for f in sorted(folder.glob("*.jpg")):
        if f.name not in data:
            data[f.name] = {"boxes": [], "by": "pending"}
            new += 1
    if new or (folder.is_dir() and not (folder / BOXES_FILE).exists()):
        save_boxes(folder, data)
    return new


# ------------------------------------------------------------ Open Images

def _save_whole(image_id, boxes, dest, image_url, opener):
    """Download one whole photo (no crop), shrink it, keep its boxes. Returns the id or None."""
    import io
    import urllib.error
    from PIL import ImageOps
    try:
        with opener(image_url.format(image_id)) as resp:
            data = resp.read()
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (urllib.error.URLError, OSError, ValueError):
        return None  # 404 / bad file
    if min(img.size) < openimages.MIN_CROP:
        return None
    img.thumbnail((DETECT_SIDE, DETECT_SIDE), Image.LANCZOS)
    img.save(dest, quality=90)
    return image_id


def download_object(name, candidates, out_dir, cap, image_url=openimages.IMAGE_URL,
                    opener=openimages.urllib.request.urlopen, workers=8, log=print):
    """Download up to `cap` whole photos for one object and write its boxes.json.

    `candidates` is {image_id: [(x0, x1, y0, y1), ...]} (Open Images order);
    photos with no boxes (keys) are saved as "pending". Returns the saved ids.
    """
    from concurrent.futures import ThreadPoolExecutor
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_boxes(out_dir)
    saved = [p.stem for p in out_dir.glob("*.jpg")]  # resume
    todo = [(i, b) for i, b in sorted(candidates.items()) if i not in set(saved)]
    new = {}
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), workers * 4):
            if len(saved) >= cap:
                break
            batch = todo[start:start + workers * 4]
            futures = [(i, b, pool.submit(_save_whole, i, b, out_dir / f"{i}.jpg",
                                          image_url, opener)) for i, b in batch]
            for i, b, fut in futures:
                if fut.result():
                    saved.append(i)
                    new[i] = b
    for extra in saved[cap:]:  # an overshoot within one batch is trimmed
        (out_dir / f"{extra}.jpg").unlink(missing_ok=True)
        new.pop(extra, None)
    saved = saved[:cap]
    for i, boxes in new.items():
        if boxes:
            data[f"{i}.jpg"] = {"by": "openimages", "boxes": [
                {"label": name, "box": [x0, y0, x1, y1]} for x0, x1, y0, y1 in boxes]}
        else:
            data[f"{i}.jpg"] = {"by": "pending", "boxes": []}
    save_boxes(out_dir, data)
    n_boxed = sum(1 for v in data.values() if v["by"] == "openimages")
    log(f"{name}: {len(saved)} whole photos, {n_boxed} with Open Images boxes "
        f"({len(saved) - n_boxed} waiting for auto-boxing)")
    return saved


def download_all(out="data/detection", cap=300, objects=None, credits=True, log=print):
    objects = {k: openimages.OBJECTS[k] for k in (objects or openimages.OBJECTS)}
    web = Path(out) / "web"
    web.mkdir(parents=True, exist_ok=True)
    cache = web / "_candidates.json"
    cands = json.loads(cache.read_text()) if cache.exists() else None
    if cands is not None and not all(o in cands for o in objects):
        cands = None
    if cands is None:
        # keys have no boxes in Open Images: whole-photo labels, boxed later (task 19)
        cands = openimages.find_candidates(cap, objects, log=log, all_boxes=True,
                                           min_box_side=MIN_BOX_SIDE)
        cands = {o: {i: (b or []) for i, b in v.items()} for o, v in cands.items()}
        cache.write_text(json.dumps(cands))
    saved = {o: download_object(o, cands[o], web / o, cap, log=log) for o in objects}
    if credits:
        openimages.write_credits(web, saved)
    return saved


# ---------------------------------------------------------- Alex's footage

def parse_distance(filename):
    """'keys_1.0m.jpg' -> ('keys', 1.0); None if the name does not follow the pattern."""
    m = DIST_NAME.match(Path(filename).stem)
    if not m:
        return None
    return m.group("obj").lower(), float(m.group("m").replace(",", "."))


def import_distance(drive_dir, root="data/detection", log=print):
    """Copy the tape-measured photos into own/distance/ with distances.json (TEST ONLY).

    Returns {"photos": n, "skipped": [names that did not match `keys_1.0m.jpg`]}.
    """
    from .objects import _open_photo, _safe
    src = Path(drive_dir) / DISTANCE
    info = {"photos": 0, "skipped": []}
    if not src.is_dir():
        log(f"{DISTANCE}: no folder at {src} (distance can't be checked yet)")
        return info
    out = Path(root) / "own" / DISTANCE
    out.mkdir(parents=True, exist_ok=True)
    table = json.loads((out / DISTANCES_FILE).read_text()) if (out / DISTANCES_FILE).exists() else {}
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or not f.is_file() or f.suffix.lower() not in PHOTO_EXTS:
            continue
        parsed = parse_distance(f.name)
        img = _open_photo(f, DETECT_SIDE) if parsed else None
        if parsed is None or img is None:
            info["skipped"].append(f.name)
            continue
        dest = f"{_safe(f.stem)}.jpg"
        if not (out / dest).exists():
            img.save(out / dest, quality=92)
        table[dest] = {"object": parsed[0], "metres": parsed[1]}
        info["photos"] += 1
    (out / DISTANCES_FILE).write_text(json.dumps(table, indent=0, sort_keys=True))
    log(f"{DISTANCE}: {info['photos']} measured photos"
        + (f"; skipped (name not like keys_1.0m.jpg): {info['skipped']}" if info["skipped"] else ""))
    return info


def import_own_detection(drive_dir, root="data/detection", fps=2.0, backend="auto", log=print):
    """Bring in Alex's footage: per-object folders, `together/` and `distance/`.

    Missing or empty folders are fine. Frames are kept at 640 px. Everything
    new is "pending" until auto-boxed. Returns a summary dict per folder.
    """
    summary = import_own(drive_dir, root, FOLDERS, fps=fps, backend=backend, log=log,
                         max_side=DETECT_SIDE)
    for name in FOLDERS:
        folder = Path(root) / "own" / name
        if folder.is_dir():
            register_pending(folder)
    summary[DISTANCE] = import_distance(drive_dir, root, log=log)
    return summary


# ------------------------------------------------------- dataset + split

@dataclass(frozen=True)
class DetSample:
    path: Path
    label: int  # index of the folder in `FOLDERS` (used only to split fairly)
    source: str  # "web" or "own"
    group: str  # image id, photo name, or video name
    folder: str  # object name or "together"
    boxes: tuple  # ((class_name, (x0, y0, x1, y1)), ...) as fractions
    by: str  # who drew the boxes


def scan_detection(root="data/detection", boxed_only=True):
    """List images that have boxes. Returns (samples, n_pending).

    `distance/` is never included (test only). Images still "pending" (not yet
    boxed) are skipped unless `boxed_only=False`, and counted.
    """
    root = Path(root)
    samples, pending = [], 0
    for source in SOURCES:
        for label, folder in enumerate(FOLDERS):
            d = root / source / folder
            for name, entry in sorted(load_boxes(d).items()):
                if not (d / name).exists():
                    continue
                if entry["by"] == "rejected":  # auto-boxer found nothing confident
                    continue
                if entry["by"] == "pending":
                    pending += 1
                    if boxed_only:
                        continue
                boxes = tuple((b["label"], tuple(b["box"])) for b in entry["boxes"])
                samples.append(DetSample(d / name, label, source, _group_of(d / name),
                                         folder, boxes, entry["by"]))
    return samples, pending


def split_detection(samples, test_frac=0.2, seed=0):
    """Train/test split by video/photo, never by frame (same rule as the classifier)."""
    return split_by_group(samples, test_frac, seed)


def load_distance_photos(root="data/detection"):
    """The test-only measured photos: [(path, object, metres)]."""
    d = Path(root) / "own" / DISTANCE
    f = d / DISTANCES_FILE
    if not f.exists():
        return []
    return [(d / n, v["object"], v["metres"]) for n, v in sorted(json.loads(f.read_text()).items())
            if (d / n).exists()]


class DetectionDataset(Dataset):
    """(image tensor 3xHxW in 0..1, target) pairs in torchvision detection format.

    target = {"boxes": float[N, 4] pixels (x0, y0, x1, y1), "labels": int64[N]},
    labels 1..3 for water_bottle / keys / shoes (0 is torchvision's background).
    """

    def __init__(self, samples, classes=OBJECT_NAMES, transform=None):
        self.samples = list(samples)
        self.classes = tuple(classes)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import numpy as np
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        w, h = img.size
        boxes = [[x0 * w, y0 * h, x1 * w, y1 * h] for _, (x0, y0, x1, y1) in s.boxes]
        labels = [self.classes.index(name) + 1 for name, _ in s.boxes]
        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        if self.transform:
            tensor, target = self.transform(tensor, target)
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def describe(samples_train, samples_test, pending=0):
    """Plain-text table: photos and boxes per folder/source, train vs test."""
    lines = [f"{'folder':<14}{'source':<6}{'train':>7}{'test':>6}{'boxes':>7}"]
    for folder in FOLDERS:
        for source in SOURCES:
            tr = [s for s in samples_train if s.folder == folder and s.source == source]
            te = [s for s in samples_test if s.folder == folder and s.source == source]
            if tr or te:
                n_boxes = sum(len(s.boxes) for s in tr + te)
                lines.append(f"{folder:<14}{source:<6}{len(tr):>7}{len(te):>6}{n_boxes:>7}")
    if pending:
        lines.append(f"({pending} images are waiting for auto-boxing and not used yet)")
    return "\n".join(lines)


def readiness_problems(samples, pending=0, objects=OBJECT_NAMES):
    """Plain-language reasons the dataset is not ready to train on ([] = ready)."""
    problems = []
    for name in objects:
        if not any(label == name for s in samples for label, _ in s.boxes):
            problems.append(f"No training photos with a {name} box. Auto-boxing (step 4) "
                            "probably failed: look for a red error above.")
    if pending:
        problems.append(f"{pending} photos are still waiting for auto-boxing, so they would be "
                        "left out. Auto-boxing (step 4) probably failed: look for a red error above.")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="whole photos + boxes from Open Images")
    d.add_argument("--out", default="data/detection")
    d.add_argument("--cap", type=int, default=300, help="max photos per object")
    d.add_argument("--objects", nargs="+", choices=list(openimages.OBJECTS), default=None)
    d.add_argument("--no-credits", action="store_true")
    i = sub.add_parser("import", help="Alex's footage: objects/, together/, distance/")
    i.add_argument("drive_dir", help="folder such as claudeknows_photos/")
    i.add_argument("--root", default="data/detection")
    i.add_argument("--fps", type=float, default=2.0)
    s = sub.add_parser("summary", help="show what is in the dataset")
    s.add_argument("--root", default="data/detection")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download_all(a.out, a.cap, a.objects, credits=not a.no_credits)
    elif a.cmd == "import":
        import_own_detection(a.drive_dir, a.root, a.fps)
    else:
        samples, pending = scan_detection(a.root)
        train, test = split_detection(samples)
        print(describe(train, test, pending))
        print(f"measured distance photos: {len(load_distance_photos(a.root))}")


if __name__ == "__main__":
    main()
