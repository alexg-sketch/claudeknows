"""Multi-object "copy-paste" augmentation (task 23b).

Open Images photos box only one kind of object (bottles in the bottle folder,
footwear in the shoe folder), so the detector rarely sees a bottle next to
shoes. This pastes 1-3 object crops taken from OTHER training photos into a
training photo and adds their boxes. Only the training split is used as a
source, so test photos never leak in. Plain torch/PIL, no extra packages.

Rules: crops are scaled 0.5x-1.5x but never taller or wider than 40% of the
photo; edges are feathered so the rectangle is not a giveaway; a paste that
overlaps an existing box by more than 10% (measured against the smaller of the
two boxes, which is stricter than IoU) is rejected.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from .objects import OBJECT_NAMES

MAX_FRACTION = 0.4
MAX_OVERLAP = 0.1
FEATHER = 5
TRIES = 10


def crop_pool(samples, classes=OBJECT_NAMES, min_side=0.03):
    """[(path, class index 1..N, (x0, y0, x1, y1) fractions)] for every box in `samples`."""
    pool = []
    for s in samples:
        for name, box in s.boxes:
            if box[2] - box[0] >= min_side and box[3] - box[1] >= min_side:
                pool.append((s.path, classes.index(name) + 1, box))
    return pool


def _overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 1.0


def _feather_mask(h, w, f=FEATHER):
    mask = torch.ones(1, h, w)
    if h > 2 * f + 2 and w > 2 * f + 2:
        mask = torch.zeros(1, h, w)
        mask[:, f:-f, f:-f] = 1.0
        mask = TF.gaussian_blur(mask, kernel_size=[2 * f + 1, 2 * f + 1], sigma=[f / 2, f / 2])
    return mask


class CompositePaste:
    """Callable (image 3xHxW 0..1, target) -> (image, target); pastes with probability `p`."""

    def __init__(self, pool, p=0.3, max_paste=3, seed=None):
        self.pool, self.p, self.max_paste = list(pool), p, max_paste
        self.rng = random.Random(seed)

    def _load(self, path, box):
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            crop = im.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))
        return torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def __call__(self, image, target):
        if not self.pool or self.rng.random() >= self.p:
            return image, target
        image = image.clone()
        boxes = [b.tolist() for b in target["boxes"]]
        labels = target["labels"].tolist()
        h, w = image.shape[-2:]
        for _ in range(self.rng.randint(1, self.max_paste)):
            path, label, box = self.rng.choice(self.pool)
            crop = self._load(path, box)
            ch, cw = crop.shape[-2:]
            scale = min(self.rng.uniform(0.5, 1.5), MAX_FRACTION * h / ch, MAX_FRACTION * w / cw)
            nh, nw = max(2, round(ch * scale)), max(2, round(cw * scale))
            if nh >= h or nw >= w:
                continue
            for _ in range(TRIES):
                x0, y0 = self.rng.randint(0, w - nw), self.rng.randint(0, h - nh)
                new = [x0, y0, x0 + nw, y0 + nh]
                if all(_overlap(new, b) <= MAX_OVERLAP for b in boxes):
                    patch = F.interpolate(crop[None], size=(nh, nw), mode="bilinear",
                                          align_corners=False)[0]
                    m = _feather_mask(nh, nw)
                    region = image[:, y0:y0 + nh, x0:x0 + nw]
                    image[:, y0:y0 + nh, x0:x0 + nw] = m * patch + (1 - m) * region
                    boxes.append([float(v) for v in new])
                    labels.append(label)
                    break
        return image, {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                       "labels": torch.tensor(labels, dtype=torch.int64)}
