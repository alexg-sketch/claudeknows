"""Object detector for water_bottle / keys / shoes (task 20).

A small pretrained torchvision detector (Faster R-CNN with a MobileNetV3
backbone) gets a new 3-object head. Training photos are WHOLE photos with
boxes, with box-aware augmentation so the model copes with any angle, size and
position. Pretrained weights download in Colab (blocked in the Claude sandbox:
tests use `pretrained=False`).

    python -m claudeknows.detector train --root data/detection --epochs 15 --out detector.pt
    python -m claudeknows.detector report --root data/detection --checkpoint detector.pt
"""
import argparse
import math
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2

from .autobox import iou
from .composite import CompositePaste, crop_pool
from .detection_data import (DetectionDataset, collate, load_distance_photos, scan_detection,
                             scan_negatives, scene_groups,
                             split_detection)
from .objects import OBJECT_NAMES

SIZE_BINS = (("far (small)", 0.0, 0.03), ("middle", 0.03, 0.15), ("near (big)", 0.15, 1.01))
MIN_BOX_PIXELS = 8
# (shortest, longest) side the model resizes photos to. Was (480, 640); bigger keeps far,
# small objects visible (run 3: 28% found far vs 87% near).
IMAGE_SIZE = (640, 960)
OLD_IMAGE_SIZE = (480, 640)  # checkpoints saved before the size was stored


# ------------------------------------------------------------------ model

def build_detector(num_objects=len(OBJECT_NAMES), pretrained=True, min_size=IMAGE_SIZE[0],
                   max_size=IMAGE_SIZE[1]):
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

def zoom_crop(img, target, rng, tries=10, keep=0.6):
    """Crop a random window (50-70% of each side, same aspect ratio) that fully holds one box.

    Boxes that stay at least `keep` (60%) inside are kept and clipped, shifted to the
    window's corner. Returns the photo and target unchanged if no window fits in `tries` draws.
    """
    h, w = img.shape[-2:]
    boxes = target["boxes"]
    if len(boxes) == 0:
        return img, target
    for _ in range(tries):
        f = rng.uniform(0.5, 0.7)
        cw, ch = max(1, int(w * f)), max(1, int(h * f))
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        win = torch.tensor([x0, y0, x0 + cw, y0 + ch], dtype=boxes.dtype)
        inter = (torch.minimum(boxes[:, 2:], win[2:]) - torch.maximum(boxes[:, :2], win[:2])).clamp(min=0)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        frac = inter[:, 0] * inter[:, 1] / area.clamp(min=1e-6)
        if not bool((frac >= 0.999).any()):
            continue
        kept = frac >= keep
        shift = torch.tensor([x0, y0, x0, y0], dtype=boxes.dtype)
        nb = (boxes[kept] - shift)
        nb[:, 0::2] = nb[:, 0::2].clamp(0, cw)
        nb[:, 1::2] = nb[:, 1::2].clamp(0, ch)
        return img[..., y0:y0 + ch, x0:x0 + cw], {**target, "boxes": nb, "labels": target["labels"][kept]}
    return img, target


def train_transform(composite=None, rot_degrees=8, rot90_p=0.15, vflip_p=0.1, zoom_p=0.0, seed=0):
    """Random flip, small rotation, exact 90-degree turns, zoom in/out, shift, perspective, colour.

    Small tilts only (`rot_degrees`, default 8): boxes get looser as the tilt grows, which
    hurts above ~3 degrees (RESEARCH.md). Phone photos are often turned a full 90 degrees,
    and those turns keep boxes exact, so they are done separately (`rot90_p` chance,
    either direction) plus a rare upside-down flip (`vflip_p`).

    Boxes follow the image. A rotated box becomes the box around the rotated
    corners (a little loose, never cut). Boxes mostly pushed out of the frame
    are dropped. `composite` (a `CompositePaste`) runs first and adds pasted objects.
    With chance `zoom_p` (default 0 = off) a zoomed window around at least one box is cut out
    first (`zoom_crop`), so objects look like they do inside a test-time tile.
    """
    zoom_rng = random.Random(seed)
    t = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=vflip_p),
        v2.RandomApply([v2.RandomChoice([v2.RandomRotation((90, 90), expand=True),
                                         v2.RandomRotation((270, 270), expand=True)])], p=rot90_p),
        v2.RandomAffine(degrees=rot_degrees, translate=(0.2, 0.2), scale=(0.5, 1.5)),
        v2.RandomPerspective(distortion_scale=0.3, p=0.3),
        v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=MIN_BOX_PIXELS, labels_getter=lambda x: x[1]["labels"]),
    ])

    def apply(img, target):
        if composite is not None:
            img, target = composite(img, target)
        if zoom_p > 0 and zoom_rng.random() < zoom_p:
            img, target = zoom_crop(img, target, zoom_rng)
        h, w = img.shape[-2:]
        empty = len(target["boxes"]) == 0
        # background-only photo: torchvision skips empty box lists when a turn changes the
        # image size, so give it a stand-in box and throw the result away afterwards
        tb = torch.tensor([[0.0, 0.0, 1.0, 1.0]]) if empty else target["boxes"]
        tl = torch.zeros(1, dtype=torch.int64) if empty else target["labels"]
        boxes = tv_tensors.BoundingBoxes(tb, format="XYXY", canvas_size=(h, w))
        img, out = t(tv_tensors.Image(img), {"boxes": boxes, "labels": tl})
        if empty:
            out = {"boxes": out["boxes"][:0], "labels": out["labels"][:0]}
        return img.as_subclass(torch.Tensor), {"boxes": out["boxes"].as_subclass(torch.Tensor),
                                               "labels": out["labels"]}

    def reseed(value):
        """Restart the transform's own random generators (data-loader workers call this)."""
        zoom_rng.seed(value)
        if composite is not None and hasattr(composite, "rng"):
            composite.rng.seed(value + 1)

    def get_state():
        return (zoom_rng.getstate(), composite.rng.getstate() if hasattr(composite, "rng") else None)

    def set_state(state):
        zoom_rng.setstate(state[0])
        if state[1] is not None:
            composite.rng.setstate(state[1])
    apply.reseed, apply.get_state, apply.set_state = reseed, get_state, set_state
    return apply


def _seed_worker(worker_id):
    """DataLoader workers inherit copies of the transform's Python random generators, so
    without this every worker (and every epoch) would repeat the same "random" draws."""
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None)
    if transform is not None and hasattr(transform, "reseed"):
        transform.reseed(torch.initial_seed() % 2 ** 31)  # differs per worker and per epoch


# ---------------------------------------------------------------- training

def mean_ap(model, samples, min_score=0.05):
    """Mean over objects of AP@0.5 on `samples` (objects with no boxes are skipped); None if no boxes."""
    results = predict_samples(model, samples, min_score)
    return _mean([average_precision(results, c) for c in OBJECT_NAMES])


def lr_factor(it, total, warmup):
    """Linear warm-up for `warmup` steps, then cosine decay to 2% of the starting rate."""
    if it < warmup:
        return (it + 1) / warmup
    progress = (it - warmup) / max(1, total - warmup)
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def train_detector(train_samples, epochs=10, batch_size=4, lr=None, pretrained=True,
                   max_batches=None, device=None, log=print, seed=0, val_samples=None,
                   patience=4, warmup=200, grad_clip=10.0, stats=None, composite=0.0,
                   rot_degrees=8, rot90_p=0.15, negatives=None, neg_frac=0.15,
                   zoom_crop_p=0.0, ema_decay=0.0, resume_path=None, workers=0):
    """Train on samples from `scan_detection`. Returns (model, history of train loss).

    Recipe (RESEARCH.md): SGD with lr 0.005 x batch/8 unless `lr` is given, linear
    warm-up then cosine decay, gradient clipping, and mixed precision on CUDA only.
    With `val_samples` (split off the train set, by video/photo) it scores mAP@0.5
    after every epoch, keeps the best epoch's weights and stops after `patience`
    epochs without improvement. `composite` is the chance (0-1) of pasting
    objects from other training photos into a photo (composite.py). `stats`, if given,
    `zoom_crop_p` is the chance of training on a zoomed window around a box (0 = off).
    `ema_decay` > 0 (e.g. 0.995) keeps a moving average of the weights, which is what gets
    validated, kept and saved (0 = off).
    `negatives` are background-only photos (no boxes); each epoch mixes in a fresh random
    pick of them, at most `neg_frac` (15%) of that epoch's photos. `stats`, if given,
    is filled with the per-epoch validation mAP ("val_map"), the best epoch and the number of epochs run.
    `resume_path`: after every epoch the whole training state is saved there (temp file, then
    rename); if the file exists and was made with the same settings, training continues from
    the next epoch (random draws are restored too, but with `workers` > 0 the photo order may
    differ from an unbroken run). The file is removed when training ends normally.
    `workers` loads photos in that many extra processes (0 = in the training process).
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.005 * batch_size / 8 if lr is None else lr
    model = build_detector(pretrained=pretrained).to(device)
    paste = CompositePaste(crop_pool(train_samples), composite, seed=seed) if composite > 0 else None
    transform = train_transform(paste, rot_degrees, rot90_p, zoom_p=zoom_crop_p, seed=seed)
    negatives = list(negatives or [])
    n_neg = min(len(negatives), round(len(train_samples) * neg_frac / (1 - neg_frac)))
    rng = random.Random(seed)
    pin = str(device).startswith("cuda") and workers > 0

    def epoch_loader():
        extra = rng.sample(negatives, n_neg) if n_neg else []
        ds = DetectionDataset(list(train_samples) + extra, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                          num_workers=workers, worker_init_fn=_seed_worker if workers else None,
                          pin_memory=pin)

    loader = epoch_loader()
    per_epoch = min(len(loader), max_batches) if max_batches else len(loader)
    total = max(1, per_epoch * epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, total, warmup))
    amp = device == "cuda" or str(device).startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    ema = None
    if ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    step = 0
    history, val_history = [], []
    best, best_epoch, best_state = -1.0, 0, None
    settings = dict(epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, composite=composite,
                    rot_degrees=rot_degrees, rot90_p=rot90_p, neg_frac=neg_frac, zoom=zoom_crop_p,
                    ema=ema_decay, max_batches=max_batches, n_train=len(train_samples),
                    n_neg=len(negatives), warmup=warmup)
    start = 0
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if ck["settings"] != settings:
            log(f"resume file {resume_path} was made with different settings: starting fresh")
        else:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            if ema is not None:
                ema.load_state_dict(ck["ema"])
            start, step = ck["epoch"], ck["step"]
            history, val_history = ck["history"], ck["val_history"]
            best, best_epoch, best_state = ck["best"], ck["best_epoch"], ck["best_state"]
            rng.setstate(ck["rng"])
            transform.set_state(ck["transform_rng"])
            torch.set_rng_state(ck["torch_rng"].cpu())
            log(f"resuming from {resume_path}: {start} of {epochs} epochs already done")
    stopped = False
    for epoch in range(start, epochs):
        model.train()
        loss_sum, n = 0.0, 0
        if epoch:
            loader = epoch_loader()
        for i, (images, targets) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = sum(model(images, targets).values())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if ema is not None:
                if step <= warmup:  # copy the raw weights until warm-up ends, then average
                    ema.module.load_state_dict(model.state_dict())
                else:
                    ema.update_parameters(model)
            loss_sum += float(loss.detach())
            n += 1
        history.append(loss_sum / max(n, 1))
        msg = f"epoch {epoch + 1}/{epochs}: loss {history[-1]:.3f}"
        if val_samples:
            v = mean_ap(ema.module if ema is not None else model, val_samples)
            v = 0.0 if v is None else v
            val_history.append(v)
            msg += f", validation mAP {100 * v:.1f}%"
            if v > best:
                best, best_epoch = v, epoch + 1
                best_state = {k: t.detach().clone()
                              for k, t in (ema.module if ema is not None else model).state_dict().items()}
            log(msg)
            if epoch + 1 - best_epoch >= patience:
                log(f"no improvement for {patience} epochs: stopping, keeping epoch {best_epoch}")
                stopped = True
        else:
            log(msg)
        if resume_path and not stopped:
            tmp = resume_path + ".tmp"
            torch.save(dict(settings=settings, model=model.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict(),
                            ema=ema.state_dict() if ema is not None else None,
                            epoch=epoch + 1, step=step, history=history, val_history=val_history,
                            best=best, best_epoch=best_epoch, best_state=best_state,
                            rng=rng.getstate(), transform_rng=transform.get_state(),
                            torch_rng=torch.get_rng_state()), tmp)
            os.replace(tmp, resume_path)
        if stopped:
            break
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    elif ema is not None:
        model.load_state_dict(ema.module.state_dict())
    if stats is not None:
        stats.update(val_map=val_history, best_epoch=best_epoch or len(history),
                     epochs_run=len(history))
    return model, history


def save_detector(model, path, history=(), extra=None):
    t = model.transform
    torch.save({"state_dict": model.state_dict(), "classes": list(OBJECT_NAMES),
                "history": list(history), "image_size": [int(t.min_size[0]), int(t.max_size)],
                **(extra or {})}, path)


def load_detector(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    min_size, max_size = ck.get("image_size", OLD_IMAGE_SIZE)
    model = build_detector(len(ck["classes"]), pretrained=False, min_size=min_size, max_size=max_size)
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


TTA_SCALE = 1.3
TILE_FRAC = 0.6  # a tile is 60% of the photo side: 2x2 tiles overlap by 20%


def _unmirror(box):
    return (1.0 - box[2], box[1], 1.0 - box[0], box[3])


def _tile_boxes():
    """The four tile windows as fractions (x0, y0, x1, y1) of the photo."""
    o = 1.0 - TILE_FRAC
    return [(x, y, x + TILE_FRAC, y + TILE_FRAC) for y in (0.0, o) for x in (0.0, o)]


def detect_tta(model, image, classes=OBJECT_NAMES, min_score=0.5, tiles=False):
    """`detect` with test-time augmentation (task 23g). Off by default everywhere.

    Runs the detector on the photo, its mirror image and a 1.3x larger version
    (the model's resize is raised for that pass, so objects really look bigger).
    With `tiles=True` it also runs on four overlapping tiles, each upscaled by
    the model's own resize, which helps small, far-away objects. All boxes are
    mapped back to the full photo and merged with weighted boxes fusion
    (`ensemble-boxes`). Plain TTA averages scores; with tiles a box seen by one
    tile alone would be averaged down below any threshold, so scores use the
    maximum instead.
    """
    from ensemble_boxes import weighted_boxes_fusion

    img = image.convert("RGB")
    w, h = img.size
    passes = [detect(model, img, classes, 0.05),
              [dict(f, box=_unmirror(f["box"]))
               for f in detect(model, img.transpose(Image.FLIP_LEFT_RIGHT), classes, 0.05)]]
    transform = getattr(model, "transform", None)
    if transform is not None:  # scale pass: make the model look at the photo larger
        old = (transform.min_size, transform.max_size)
        transform.min_size = tuple(int(m * TTA_SCALE) for m in old[0])
        transform.max_size = int(old[1] * TTA_SCALE)
        try:
            passes.append(detect(model, img, classes, 0.05))
        finally:
            transform.min_size, transform.max_size = old
    if tiles:
        for x0, y0, x1, y1 in _tile_boxes():
            tile = img.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
            tw, th = (x1 - x0), (y1 - y0)
            passes.append([dict(f, box=(x0 + f["box"][0] * tw, y0 + f["box"][1] * th,
                                        x0 + f["box"][2] * tw, y0 + f["box"][3] * th))
                           for f in detect(model, tile, classes, 0.05)])
    clip = lambda b: [min(1.0, max(0.0, v)) for v in b]
    boxes = [[clip(f["box"]) for f in p] or [[0, 0, 0, 0]] for p in passes]
    scores = [[f["score"] for f in p] or [0.0] for p in passes]
    labels = [[classes.index(f["label"]) for f in p] or [0] for p in passes]
    b, sc, lb = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.55, skip_box_thr=0.05,
                                      conf_type="max" if tiles else "avg")
    found = [{"label": classes[int(l)], "score": float(s), "box": tuple(float(v) for v in bx)}
             for bx, s, l in zip(b, sc, lb) if float(s) >= min_score and bx[2] > bx[0] and bx[3] > bx[1]]
    return sorted(found, key=lambda f: -f["score"])


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


def predict_samples(model, samples, min_score=0.05, tta=None):
    """Run the detector on test samples. Returns [(sample, gt, preds)] for `average_precision`.

    `tta` is None (plain), "flip" (original + mirror + 1.3x) or "tiles" (also 4 tiles)."""
    out = []
    for s in samples:
        with Image.open(s.path) as im:
            found = (detect_tta(model, im, min_score=min_score, tiles=(tta == "tiles"))
                     if tta else detect(model, im, min_score=min_score))
        out.append((s, list(s.boxes), [(f["label"], f["score"], f["box"]) for f in found]))
    return out


def false_alarms_per_photo(model, negatives, min_score=0.5):
    """Average number of objects the detector claims in background-only photos (lower is better).

    None if there are no negative photos."""
    if not negatives:
        return None
    total = 0
    for s in negatives:
        with Image.open(s.path) as im:
            total += len(detect(model, im, min_score=min_score))
    return total / len(negatives)


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


def _group_key(sample):
    """Photos from one video count as one unit; an internet photo is its own unit."""
    return ("own", sample.group) if sample.source == "own" else ("web", str(sample.path))


def _boot_scores(results, classes, n, seed, paired_with=None):
    """n resampled scores (mean AP over objects, and per object); whole videos/photos are resampled."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keys = sorted({_group_key(s) for s, _, _ in results})
    by_key = defaultdict(list)
    for i, (s, _, _) in enumerate(results):
        by_key[_group_key(s)].append(i)
    out = []
    for _ in range(n):
        picked = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        idx = [i for k in picked for i in by_key[k]]
        sets = [[results[i] for i in idx]]
        if paired_with is not None:
            sets.append([paired_with[i] for i in idx])
        out.append([{c: average_precision(r, c) for c in classes} for r in sets])
    return out


def bootstrap_ci(results, classes=OBJECT_NAMES, n=1000, seed=0):
    """95% range (2.5-97.5%) of mAP@0.5, overall and per object, by resampling whole videos/photos.

    Returns {"overall" or object name: (low, high)}; None where an object has no boxes in a draw.
    Few videos means a wide range: that is the honest answer about how much to trust a score.
    """
    import numpy as np
    draws = [d[0] for d in _boot_scores(results, classes, n, seed)]
    ci = {}
    for name in ("overall", *classes):
        vals = [_mean(list(d.values())) if name == "overall" else d[name] for d in draws]
        vals = [v for v in vals if v is not None]
        ci[name] = tuple(float(x) for x in np.percentile(vals, [2.5, 97.5])) if vals else None
    return ci


def compare_runs(results_a, results_b, classes=OBJECT_NAMES, n=1000, seed=0):
    """Paired A/B verdict on overall mAP@0.5 for two runs over the SAME test samples (same order).

    Returns (difference b - a, (low, high), verdict) where verdict is "helped",
    "hurt" or "no clear difference" (the range of the difference excludes 0 or not).
    """
    import numpy as np
    assert [s.path for s, _, _ in results_a] == [s.path for s, _, _ in results_b], \
        "runs must cover the same samples"
    diffs = []
    for a, b in _boot_scores(results_a, classes, n, seed, paired_with=results_b):
        ma, mb = _mean(list(a.values())), _mean(list(b.values()))
        if ma is not None and mb is not None:
            diffs.append(mb - ma)
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    full_a = _mean([average_precision(results_a, c) for c in classes]) or 0.0
    full_b = _mean([average_precision(results_b, c) for c in classes]) or 0.0
    verdict = "helped" if lo > 0 else "hurt" if hi < 0 else "no clear difference"
    return full_b - full_a, (lo, hi), verdict


def format_report(report, ci=None, false_alarms=None, test_scenes=None):
    """`ci` (from `bootstrap_ci`) adds the 95% range next to the overall and per-object scores.

    `false_alarms` (from `false_alarms_per_photo`) and `test_scenes` (number of different
    places in Alex's test footage) are added as extra lines when given."""
    lines = []
    for sec, rows in report.items():
        lines.append(f"\n{sec}")
        for name, (ap, n) in rows.items():
            score = "  n/a " if ap is None else f"{100 * ap:5.1f}%"
            key = "overall" if sec == "overall" else name if sec == "per object" else None
            rng = ci.get(key) if ci and key else None
            extra = f"  (range {100 * rng[0]:.0f}-{100 * rng[1]:.0f}%)" if rng else ""
            lines.append(f"  {name:<28}{score}{extra}   ({n} boxes)")
    if test_scenes is not None:
        lines.append(f"\nown footage in the test set comes from {test_scenes} different scene(s)")
    if false_alarms is not None:
        lines.append(f"false alarms per background-only photo (score 0.5): {false_alarms:.2f}")
    return "\n".join(lines).lstrip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="data/detection")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=None,
                   help="default 0.005 x batch/8")
    t.add_argument("--val-frac", type=float, default=0.15,
                   help="share of the training photos/videos held back to pick the best epoch")
    t.add_argument("--patience", type=int, default=4)
    t.add_argument("--composite", type=float, default=0.0,
                   help="chance (0-1) of pasting 1-3 objects from other training photos into a photo")
    t.add_argument("--rot-degrees", type=float, default=8,
                   help="largest random tilt in degrees (was 20; small tilts keep boxes tight)")
    t.add_argument("--rot90-p", type=float, default=0.15,
                   help="chance of an exact 90-degree turn")
    t.add_argument("--neg-frac", type=float, default=0.15,
                   help="largest share of each epoch that is background-only photos (0 = none)")
    t.add_argument("--zoom-crop-p", type=float, default=0.0,
                   help="chance of training on a zoomed window around a box (pairs with --tta tiles)")
    t.add_argument("--ema-decay", type=float, default=0.0,
                   help="moving average of the weights, e.g. 0.995 (0 = off)")
    t.add_argument("--resume", default=None, metavar="PATH",
                   help="save progress here after every epoch and continue from it if it exists")
    t.add_argument("--workers", type=int, default=2,
                   help="extra processes that load photos (0 = none)")
    t.add_argument("--max-batches", type=int, default=None)
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--out", default="detector.pt")
    r = sub.add_parser("report")
    r.add_argument("--root", default="data/detection")
    r.add_argument("--checkpoint", default="detector.pt")
    r.add_argument("--tta", choices=["flip", "tiles"], default=None,
                   help="test-time augmentation: flip = mirror + 1.3x, tiles = also 4 zoomed tiles")
    a = p.parse_args(argv)
    samples, pending = scan_detection(a.root)
    scenes = scene_groups(samples)
    train, test = split_detection(samples, scenes=scenes)
    negs_train, negs_test = split_detection(scan_negatives(a.root))
    if a.cmd == "train":
        fit, val = (split_detection(train, a.val_frac, scenes=scenes) if a.val_frac > 0
                    else (train, []))
        stats = {}
        model, hist = train_detector(fit, a.epochs, a.batch_size, a.lr, not a.no_pretrained,
                                     a.max_batches, val_samples=val or None, patience=a.patience,
                                     stats=stats, composite=a.composite,
                                     rot_degrees=a.rot_degrees, rot90_p=a.rot90_p,
                                     negatives=negs_train if a.neg_frac > 0 else None,
                                     neg_frac=a.neg_frac, zoom_crop_p=a.zoom_crop_p,
                                     ema_decay=a.ema_decay, resume_path=a.resume,
                                     workers=a.workers)
        save_detector(model, a.out, hist, extra={"train_stats": stats})
        print(f"saved {a.out} (best epoch {stats['best_epoch']} of {stats['epochs_run']} run)")
    else:
        model, classes = load_detector(a.checkpoint)
        results = predict_samples(model, test, tta=a.tta)
        alarms = false_alarms_per_photo(model, negs_test)
        n_scenes = len({scenes[x.group] for x in test if x.source == "own"}) or None
        print(format_report(build_report(results, classes), bootstrap_ci(results, classes),
                            alarms, n_scenes))
        if a.tta:
            plain = predict_samples(model, test)
            print(f"\nwithout --tta {a.tta}:")
            print(format_report(build_report(plain, classes), bootstrap_ci(plain, classes)))
        print(f"\n(distance photos, used in task 21: {len(load_distance_photos(a.root))})")


if __name__ == "__main__":
    main()
