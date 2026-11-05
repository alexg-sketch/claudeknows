"""Auto-boxing: draw boxes on photos nobody has boxed yet (task 19).

Open Images has no boxes for keys, and Alex's footage has none at all. An
open-vocabulary detector (OWLv2, Apache-2.0, via `transformers`) finds objects
from text prompts. Only confident boxes are kept, and a contact sheet lets a
human glance over the result.

    python -m claudeknows.autobox run data/detection --threshold 0.3
    python -m claudeknows.autobox sheet data/detection/own/keys --out keys_sheet.jpg

OWLv2 downloads from Hugging Face, which is blocked in the Claude sandbox, so
tests pass a fake `detect` function. Colab runs the real one.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from .detection_data import FOLDERS, SOURCES, TOGETHER, load_boxes, save_boxes
from .objects import OBJECT_NAMES

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
PROMPTS = {"water_bottle": "a water bottle", "keys": "a bunch of keys", "shoes": "a shoe"}
THRESHOLD = 0.3
NMS_IOU = 0.5
EXTRA = "owlv2-extra"  # `by` of a box added to a photo that already had boxes (task 23c)
FILL_THRESHOLD = 0.4
FILL_IOU = 0.3
REJECTED = "rejected"  # looked at, nothing confident found: not used for training
COLOURS = {"water_bottle": "#1f77b4", "keys": "#d62728", "shoes": "#2ca02c"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(found, iou_thresh=NMS_IOU):
    """found: [(label, score, box)]. Keep the best of overlapping same-label boxes."""
    keep = []
    for f in sorted(found, key=lambda f: -f[1]):
        if all(k[0] != f[0] or iou(k[2], f[2]) < iou_thresh for k in keep):
            keep.append(f)
    return keep


def post_process(processor, outputs, threshold, target_sizes):
    """OWLv2 boxes from raw outputs. transformers 5 renamed the processor's
    `post_process_object_detection` to `post_process_grounded_object_detection`."""
    fn = (getattr(processor, "post_process_grounded_object_detection", None)
          or processor.post_process_object_detection)
    return fn(outputs, threshold=threshold, target_sizes=target_sizes)


def owl_detector(model_name=OWL_MODEL, device=None):
    """Load OWLv2 (Colab only). Returns detect(image, labels) -> [(label, score, (x0,y0,x1,y1) fractions)]."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

    def detect(image, labels, threshold=0.1):
        w, h = image.size
        side = max(w, h)  # OWLv2 pads to a square: boxes are relative to that square
        texts = [[PROMPTS[l] for l in labels]]
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = post_process(processor, out, threshold, torch.tensor([[side, side]], device=device))[0]
        found = []
        for score, idx, box in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box)
            found.append((labels[int(idx)], float(score),
                          (max(0, x0 / w), max(0, y0 / h), min(1, x1 / w), min(1, y1 / h))))
        return found

    return detect


def autobox_folder(folder, detect, expect=None, threshold=THRESHOLD, log=print):
    """Box every "pending" photo in `folder`. Returns {"boxed": n, "rejected": n, "boxes": n}.

    `expect` is the one object a single-object folder holds (only that label is
    kept: a shoe in the corner of a keys video is not a keys box); None keeps
    all three (the `together/` folder). Photos with no confident box become
    "rejected" so they are skipped, not retried every run.
    """
    folder = Path(folder)
    data = load_boxes(folder)
    labels = [expect] if expect else list(OBJECT_NAMES)
    stats = {"boxed": 0, "rejected": 0, "boxes": 0}
    for name, entry in sorted(data.items()):
        if entry["by"] != "pending" or not (folder / name).exists():
            continue
        with Image.open(folder / name) as im:
            found = detect(im.convert("RGB"), labels)
        found = nms([f for f in found if f[1] >= threshold and f[0] in labels])
        if found:
            entry["by"] = "owlv2"
            entry["boxes"] = [{"label": l, "box": [round(v, 4) for v in b], "score": round(s, 3)}
                              for l, s, b in found]
            stats["boxed"] += 1
            stats["boxes"] += len(found)
        else:
            entry["by"] = REJECTED
            entry["boxes"] = []
            stats["rejected"] += 1
    save_boxes(folder, data)
    log(f"{folder.name}: {stats['boxed']} photos boxed ({stats['boxes']} boxes), "
        f"{stats['rejected']} with nothing confident")
    return stats


def autobox_all(root, detect, threshold=THRESHOLD, log=print):
    """Auto-box all pending photos under data/detection/{web,own}/<folder>."""
    out = {}
    for source in SOURCES:
        for folder in FOLDERS:
            d = Path(root) / source / folder
            if d.is_dir() and any(v["by"] == "pending" for v in load_boxes(d).values()):
                out[f"{source}/{folder}"] = autobox_folder(
                    d, detect, None if folder == TOGETHER else folder, threshold, log)
    return out


def fill_missing(folder, detect, threshold=FILL_THRESHOLD, max_iou=FILL_IOU, log=print):
    """Add boxes for objects a photo has no box for yet (task 23c).

    Open Images only boxes the one object a photo was picked for, so a photo of
    a bottle with shoes in the background teaches the detector that those shoes
    are "nothing". Run OWLv2 with all three prompts and append a box
    (`"by": "owlv2-extra"`) only for a class the photo does not already have,
    with score >= `threshold` and IoU < `max_iou` against every existing box.
    Existing boxes (human or OWLv2) are never touched. Photos that are pending
    or rejected are skipped; each photo is checked once (`"filled": true`).
    Returns {"photos": n, "added": n}.
    """
    folder = Path(folder)
    data = load_boxes(folder)
    stats = {"photos": 0, "added": 0}
    for name, entry in sorted(data.items()):
        if entry["by"] in ("pending", REJECTED) or entry.get("filled") or not (folder / name).exists():
            continue
        with Image.open(folder / name) as im:
            found = detect(im.convert("RGB"), list(OBJECT_NAMES))
        have = {b["label"] for b in entry["boxes"]}
        existing = [b["box"] for b in entry["boxes"]]
        added = []
        for label, score, box in nms([f for f in found if f[1] >= threshold]):
            if label in have or any(iou(box, e) >= max_iou for e in existing + [a["box"] for a in added]):
                continue
            added.append({"label": label, "box": [round(v, 4) for v in box],
                          "score": round(score, 3), "by": EXTRA})
        entry["boxes"] = entry["boxes"] + added
        entry["filled"] = True
        stats["photos"] += bool(added)
        stats["added"] += len(added)
    save_boxes(folder, data)
    log(f"{folder.parent.name}/{folder.name}: added {stats['added']} missing boxes to {stats['photos']} photos")
    return stats


def fill_all(root, detect, threshold=FILL_THRESHOLD, log=print):
    """`fill_missing` on every web/own folder, including human-boxed ones."""
    out = {}
    for source in SOURCES:
        for folder in FOLDERS:
            d = Path(root) / source / folder
            if d.is_dir():
                out[f"{source}/{folder}"] = fill_missing(d, detect, threshold, log=log)
    return out


def contact_sheet(folder, out=None, n=24, cols=6, thumb=200, only_by=("owlv2",), seed=0):
    """Grid of photos with their boxes drawn, for a quick human look. Returns the image."""
    import random
    folder = Path(folder)
    items = [(k, v) for k, v in sorted(load_boxes(folder).items())
             if (v["by"] in only_by or any(b.get("by") == EXTRA for b in v["boxes"]))
             and (folder / k).exists()]
    random.Random(seed).shuffle(items)
    items = items[:n]
    rows = max(1, -(-len(items) // cols))
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, (name, entry) in enumerate(items):
        with Image.open(folder / name) as im:
            im = im.convert("RGB")
        im.thumbnail((thumb, thumb))
        draw = ImageDraw.Draw(im)
        w, h = im.size
        for b in entry["boxes"]:
            x0, y0, x1, y1 = b["box"]
            draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=COLOURS[b["label"]], width=3)
            draw.text((x0 * w + 3, y0 * h + 2), f"{b['label']} {b.get('score', ''):}", fill="white")
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    if out:
        sheet.save(out, quality=88)
    return sheet


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="auto-box all pending photos (needs transformers; Colab)")
    r.add_argument("root", nargs="?", default="data/detection")
    r.add_argument("--threshold", type=float, default=THRESHOLD)
    f = sub.add_parser("fill", help="add boxes for objects a photo has no box for (needs transformers; Colab)")
    f.add_argument("root", nargs="?", default="data/detection")
    f.add_argument("--threshold", type=float, default=FILL_THRESHOLD)
    s = sub.add_parser("sheet", help="contact sheet of one folder's auto-boxes")
    s.add_argument("folder")
    s.add_argument("--out", default="contact_sheet.jpg")
    s.add_argument("--n", type=int, default=24)
    a = p.parse_args(argv)
    if a.cmd == "run":
        autobox_all(a.root, owl_detector(), a.threshold)
    elif a.cmd == "fill":
        fill_all(a.root, owl_detector(), a.threshold)
    else:
        contact_sheet(a.folder, a.out, a.n)
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
