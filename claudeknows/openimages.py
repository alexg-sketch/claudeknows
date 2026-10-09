"""Download training photos for water bottles, keys and shoes from Google's Open Images.

    python -m claudeknows.openimages --out data/objects --cap 300

Photos land in `<out>/web/<object>/<ImageID>.jpg` (small JPEGs, cropped to the
object's bounding box where Open Images has one). A `CREDITS.md` (photographer
+ licence, CC BY) is written next to them. Alex's own photos/videos go in
`<out>/own/<object>/` (see `claudeknows.objects`).
"""
import argparse
import csv
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

BASE = "https://storage.googleapis.com/openimages"
LABELS_URL = f"{BASE}/v7/oidv7-train-annotations-human-imagelabels.csv"
BOXES_URL = f"{BASE}/v6/oidv6-train-annotations-bbox.csv"  # v7's box file is not public
META_URL = f"{BASE}/v6/oidv6-train-images-with-labels-with-rotation.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"

# label = whole-photo label(s); box = bounding-box label to crop to (None = whole photo).
#   label + box : photo must carry the label AND have a box (most photos have the
#                 object tiny, so we crop to the box)
#   box only    : any photo with a box of that kind
#   label only  : whole photo
OBJECTS = {
    "water_bottle": {"labels": ["/m/0118n_9r"], "box": "/m/04dr76w"},  # Bottle
    "keys": {"labels": ["/m/03v3yw", "/m/03lnq3"], "box": None},  # Key, Keychain
    "shoes": {"labels": [], "box": "/m/09j5n"},  # Footwear
}
OVERSAMPLE = 4  # candidates per wanted photo (many ids 404 or are too small)
MIN_CROP = 64  # skip crops smaller than this many pixels on the short side
MAX_SIDE = 256
MARGIN = 0.10  # extra context around a box


def _lines(url, opener=urllib.request.urlopen):
    """Stream a (possibly huge) CSV as text lines without loading it all."""
    with opener(url) as resp:
        yield from io.TextIOWrapper(resp, encoding="utf-8", newline="")


def find_candidates(cap, objects=OBJECTS, labels_url=LABELS_URL, boxes_url=BOXES_URL,
                    opener=urllib.request.urlopen, log=print):
    """Return {object: {image_id: box or None}} (box = (x0, x1, y0, y1) in 0..1)."""
    want = cap * OVERSAMPLE
    label_to_obj = {l: o for o, c in objects.items() for l in c["labels"]}
    labelled = {o: set() for o in objects}
    label_only = [o for o, c in objects.items() if c["labels"] and not c["box"]]
    needs_all_labels = any(c["labels"] and c["box"] for c in objects.values())
    # Pass 1: whole-photo labels. File is sorted by ImageID, so objects that
    # only need "enough" photos can stop early; ones needing every id cannot.
    if label_to_obj:
        log("scanning the Open Images label file (large; takes a minute)...")
        for line in _lines(labels_url, opener):
            if not line.rstrip().endswith(",1.0"):
                continue
            image_id, _, label, _ = line.rstrip().split(",")
            obj = label_to_obj.get(label)
            if obj:
                labelled[obj].add(image_id)
            if label_only and not needs_all_labels and all(
                    len(labelled[o]) >= want for o in label_only):
                break
    # Pass 2: bounding boxes.
    boxes = {o: {} for o, c in objects.items() if c["box"]}
    if boxes:
        log("scanning the Open Images box file (large; takes a minute)...")
        box_to_obj = {objects[o]["box"]: o for o in boxes}
        needs_full = any(objects[o]["labels"] for o in boxes)
        for row in csv.reader(_lines(boxes_url, opener)):
            obj = box_to_obj.get(row[2]) if len(row) > 11 else None
            if obj is None or row[0] == "ImageID":
                continue
            if row[10] == "1" or row[11] == "1":  # group of objects / drawing
                continue
            if objects[obj]["labels"] and row[0] not in labelled[obj]:
                continue
            x0, x1, y0, y1 = (float(row[i]) for i in (4, 5, 6, 7))
            area = (x1 - x0) * (y1 - y0)
            old = boxes[obj].get(row[0])
            if old is None or area > (old[1] - old[0]) * (old[3] - old[2]):
                boxes[obj][row[0]] = (x0, x1, y0, y1)
            if not needs_full and all(len(b) >= want for b in boxes.values()):
                break
    out = {}
    for o, c in objects.items():
        if c["box"]:
            out[o] = dict(boxes[o])
        else:
            out[o] = {i: None for i in sorted(labelled[o])}
    return out


def crop_and_shrink(img, box):
    """Crop to the box (+margin) and shrink; returns None if the crop is too small."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    if box:
        x0, x1, y0, y1 = box
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        px = (max(0, (x0 - mx) * w), max(0, (y0 - my) * h),
              min(w, (x1 + mx) * w), min(h, (y1 + my) * h))
        img = img.crop(tuple(round(v) for v in px))
    if min(img.size) < MIN_CROP:
        return None
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def _fetch(image_id, box, dest, image_url, opener):
    try:
        with opener(image_url.format(image_id)) as resp:
            data = resp.read()
        img = crop_and_shrink(Image.open(io.BytesIO(data)), box)
    except (urllib.error.URLError, OSError, ValueError):
        return None  # 404 / bad file
    if img is None:
        return None
    img.save(dest, quality=90)
    return image_id


def download_object(name, candidates, out_dir, cap, image_url=IMAGE_URL,
                    opener=urllib.request.urlopen, workers=8, log=print):
    """Download up to `cap` photos for one object. Returns the saved image ids."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = [p.stem for p in out_dir.glob("*.jpg")]  # resume
    todo = [(i, b) for i, b in sorted(candidates.items()) if i not in set(saved)]
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), workers * 4):
            if len(saved) >= cap:
                break
            batch = todo[start:start + workers * 4]
            futures = [pool.submit(_fetch, i, b, out_dir / f"{i}.jpg", image_url, opener)
                       for i, b in batch]
            saved += [r for r in (f.result() for f in futures) if r]
    # an overshoot within one batch is trimmed so the cap holds exactly
    for extra in saved[cap:]:
        (out_dir / f"{extra}.jpg").unlink(missing_ok=True)
    saved = saved[:cap]
    log(f"{name}: {len(saved)} photos (from {len(todo)} untried candidates)")
    return saved


def write_credits(web_dir, saved, meta_url=META_URL, opener=urllib.request.urlopen):
    """Write CREDITS.md with photographer + licence for every saved photo."""
    ids = {i for v in saved.values() for i in v}
    found = {}
    for row in csv.reader(_lines(meta_url, opener)):
        if row and row[0] in ids and len(row) > 7:
            found[row[0]] = row
            if len(found) == len(ids):
                break
    lines = ["# Photo credits", "",
             "Photos from Google's Open Images dataset, used under Creative "
             "Commons licences (CC BY). Each photo is named by its Open Images ID.", ""]
    for obj, ids_ in saved.items():
        lines += [f"## {obj}", ""]
        for i in sorted(ids_):
            r = found.get(i)
            if r:
                lines.append(f"- `{i}.jpg`: \"{r[7]}\" by [{r[6]}]({r[5]}), "
                             f"[licence]({r[4]}), [source]({r[3]})")
            else:
                lines.append(f"- `{i}.jpg`: Open Images ID {i} (credit not found)")
        lines.append("")
    path = Path(web_dir) / "CREDITS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def download_all(out="data/objects", cap=300, objects=None, credits=True, log=print):
    objects = {k: OBJECTS[k] for k in (objects or OBJECTS)}
    web = Path(out) / "web"
    web.mkdir(parents=True, exist_ok=True)
    cache = web / "_candidates.json"
    if cache.exists():
        cands = json.loads(cache.read_text())
        if not all(o in cands for o in objects):
            cands = None
    else:
        cands = None
    if cands is None:
        cands = find_candidates(cap, objects, log=log)
        cache.write_text(json.dumps(cands))
    saved = {o: download_object(o, cands[o], web / o, cap, log=log) for o in objects}
    if credits:
        write_credits(web, saved)
    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/objects")
    p.add_argument("--cap", type=int, default=300, help="max photos per object")
    p.add_argument("--objects", nargs="+", choices=list(OBJECTS), default=None)
    p.add_argument("--no-credits", action="store_true")
    a = p.parse_args(argv)
    download_all(a.out, a.cap, a.objects, credits=not a.no_credits)


if __name__ == "__main__":
    main()
