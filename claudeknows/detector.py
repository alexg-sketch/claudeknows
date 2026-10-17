"""Object detector for water_bottle / keys / shoes (task 20).

A small pretrained torchvision detector (Faster R-CNN with a MobileNetV3
backbone) gets a new 3-object head. Training photos are WHOLE photos with
boxes, with box-aware augmentation so the model copes with any angle, size and
position. Pretrained weights download in Colab (blocked in the Claude sandbox:
tests use `pretrained=False`).

    python -m claudeknows.detector train --root data/detection --epochs 10 --out detector.pt
    python -m claudeknows.detector report --root data/detection --checkpoint detector.pt
"""
import argparse
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

from .autobox import iou
from .detection_data import (DetectionDataset, collate, load_distance_photos, scan_detection,
                             split_detection)
from .objects import OBJECT_NAMES

SIZE_BINS = (("far (small)", 0.0, 0.03), ("middle", 0.03, 0.15), ("near (big)", 0.15, 1.01))
MIN_BOX_PIXELS = 8


# ------------------------------------------------------------------ model

def build_detector(num_objects=len(OBJECT_NAMES), pretrained=True, min_size=480, max_size=640):
    """Faster R-CNN MobileNetV3-Large FPN; class 0 is background, 1..N the objects."""
    kwargs = dict(min_size=min_size, max_size=max_size)
    if pretrained:
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, **kwargs)
    else:
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_objects + 1)
    return model


# ----------------------------------------------------- box-aware augmentation

def train_transform():
    """Random flip, rotation up to 20 degrees, zoom in/out, shift, perspective, colour.

    Boxes follow the image. A rotated box becomes the box around the rotated
    corners (a little loose, never cut). Boxes mostly pushed out of the frame
    are dropped.
    """
    t = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5)),
        v2.RandomPerspective(distortion_scale=0.3, p=0.3),
        v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=MIN_BOX_PIXELS, labels_getter=lambda x: x[1]["labels"]),
    ])

    def apply(img, target):
        h, w = img.shape[-2:]
        boxes = tv_tensors.BoundingBoxes(target["boxes"], format="XYXY", canvas_size=(h, w))
        img, out = t(tv_tensors.Image(img), {"boxes": boxes, "labels": target["labels"]})
        return img.as_subclass(torch.Tensor), {"boxes": out["boxes"].as_subclass(torch.Tensor),
                                               "labels": out["labels"]}
    return apply


# ---------------------------------------------------------------- training

def train_detector(train_samples, epochs=10, batch_size=4, lr=1e-3, pretrained=True,
                   max_batches=None, device=None, log=print, seed=0):
    """Train on samples from `scan_detection`. Returns (model, history)."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(pretrained=pretrained).to(device)
    ds = DetectionDataset(train_samples, transform=train_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for i, (images, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        history.append(total / max(n, 1))
        log(f"epoch {epoch + 1}/{epochs}: loss {history[-1]:.3f}")
    return model, history


def save_detector(model, path, history=(), extra=None):
    torch.save({"state_dict": model.state_dict(), "classes": list(OBJECT_NAMES),
                "history": list(history), **(extra or {})}, path)


def load_detector(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_detector(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["classes"]


@torch.no_grad()
def detect(model, image, classes=OBJECT_NAMES, min_score=0.5):
    """Find objects in a PIL image. Returns [{"label", "score", "box": (x0,y0,x1,y1) fractions}]."""
    import numpy as np
    device = next(model.parameters()).device
    model.eval()
    img = image.convert("RGB")
    w, h = img.size
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
    out = model([t])[0]
    return [{"label": classes[int(l) - 1], "score": float(s),
             "box": (float(b[0]) / w, float(b[1]) / h, float(b[2]) / w, float(b[3]) / h)}
            for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu())
            if float(s) >= min_score]


# -------------------------------------------------------------- evaluation

def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def size_bin(box):
    """Near/far from how much of the photo the box fills."""
    a = box_area(box)
    return next(name for name, lo, hi in SIZE_BINS if lo <= a < hi)


def position_bin(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return "centre" if 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75 else "edge"


def average_precision(images, label, keep=lambda box, sample: True, iou_thresh=0.5):
    """AP@0.5 (area under the precision/recall curve) for one object.

    images: [(sample, gt, preds)] with gt [(label, box)] and preds [(label, score, box)]
    in fractions. `keep(box, sample)` restricts the score to a subset (e.g. far
    objects): ground truths outside it are ignored, as are predictions that
    match them. Returns None when the subset has no ground-truth boxes.
    """
    scored, n_gt = [], 0
    for sample, gt, preds in images:
        g = [b for l, b in gt if l == label]
        n_gt += sum(1 for b in g if keep(b, sample))
        used = set()
        for l, s, b in sorted((p for p in preds if p[0] == label), key=lambda p: -p[1]):
            best, best_iou = None, iou_thresh
            for j, gb in enumerate(g):
                if j not in used and iou(b, gb) >= best_iou:
                    best, best_iou = j, iou(b, gb)
            if best is not None:
                used.add(best)
                if keep(g[best], sample):
                    scored.append((s, 1))
            elif keep(b, sample):
                scored.append((s, 0))
    if n_gt == 0:
        return None
    scored.sort(key=lambda x: -x[0])
    tp = fp = 0
    points = []
    for _, hit in scored:
        tp, fp = tp + hit, fp + (1 - hit)
        points.append((tp / n_gt, tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for i, (recall, _) in enumerate(points):
        precision = max(p for _, p in points[i:])  # monotone precision envelope
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def predict_samples(model, samples, min_score=0.05):
    """Run the detector on test samples. Returns [(sample, gt, preds)] for `average_precision`."""
    out = []
    for s in samples:
        with Image.open(s.path) as im:
            found = detect(model, im, min_score=min_score)
        out.append((s, list(s.boxes), [(f["label"], f["score"], f["box"]) for f in found]))
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_report(results, classes=OBJECT_NAMES):
    """mAP@0.5 overall and broken down by source, object size, position and video.

    Returns {section: {row name: (mAP or None, number of ground-truth boxes)}}.
    """
    def n_boxes(keep):
        return sum(1 for s, gt, _ in results for _, b in gt if keep(b, s))

    def section(keep):
        return _mean([average_precision(results, c, keep) for c in classes]), n_boxes(keep)

    report = {"overall": {"all": section(lambda b, s: True)},
              "per object": {c: (average_precision(results, c),
                                 sum(1 for _, gt, _ in results for l, _ in gt if l == c))
                             for c in classes},
              "by source": {}, "by size in photo": {}, "by position": {}, "by video/photo": {}}
    for src in sorted({s.source for s, _, _ in results}):
        report["by source"][src] = section(lambda b, s, src=src: s.source == src)
    for name, _, _ in SIZE_BINS:
        report["by size in photo"][name] = section(lambda b, s, name=name: size_bin(b) == name)
    for pos in ("centre", "edge"):
        report["by position"][pos] = section(lambda b, s, pos=pos: position_bin(b) == pos)
    groups = defaultdict(int)
    for s, _, _ in results:
        if s.source == "own":
            groups[s.group] += 1
    for g in sorted(groups):
        report["by video/photo"][g] = section(lambda b, s, g=g: s.group == g)
    return report


def format_report(report):
    lines = []
    for sec, rows in report.items():
        lines.append(f"\n{sec}")
        for name, (ap, n) in rows.items():
            score = "  n/a " if ap is None else f"{100 * ap:5.1f}%"
            lines.append(f"  {name:<28}{score}   ({n} boxes)")
    return "\n".join(lines).lstrip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="data/detection")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--max-batches", type=int, default=None)
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--out", default="detector.pt")
    r = sub.add_parser("report")
    r.add_argument("--root", default="data/detection")
    r.add_argument("--checkpoint", default="detector.pt")
    a = p.parse_args(argv)
    samples, pending = scan_detection(a.root)
    train, test = split_detection(samples)
    if a.cmd == "train":
        model, hist = train_detector(train, a.epochs, a.batch_size, a.lr, not a.no_pretrained,
                                     a.max_batches)
        save_detector(model, a.out, hist)
        print(f"saved {a.out}")
    else:
        model, classes = load_detector(a.checkpoint)
        print(format_report(build_report(predict_samples(model, test), classes)))
        print(f"\n(distance photos, used in task 21: {len(load_distance_photos(a.root))})")


if __name__ == "__main__":
    main()
