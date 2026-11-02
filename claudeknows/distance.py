"""How far away is each object? (task 21)

No training. Two independent estimates per detected box:

1. Depth Anything V2 **Small** (Apache-2.0; Base/Large are non-commercial, never use them):
   the relative model gives near/far ORDER, the metric-indoor Small model gives rough metres.
2. A size prior: `focal_px * real_size_m / longest_box_side_px`, from the photo's EXIF
   focal length and the typical real size of the object. Only possible when EXIF exists
   (phone photos yes, video frames and downloaded photos no).

Where the two metre estimates disagree a lot the output says so. Near/far order is
reliable, metres are approximate. Alex's tape-measured `distance/` photos are test-only:
a scale correction for the metric model is fitted leave-one-out, never on the photo it scores.

    python -m claudeknows.distance check --root data/detection --checkpoint detector.pt

The depth models download from Hugging Face (blocked in the Claude sandbox), so tests pass
fake depth functions. Colab runs the real ones.
"""
import argparse
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

RELATIVE_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
METRIC_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
# longest side of the object in metres (keys vary most: low confidence)
REAL_SIZE_M = {"water_bottle": 0.22, "shoes": 0.28, "keys": 0.08}
LOW_CONFIDENCE = {"keys"}
# weight of the size estimate in the log-space blend of the two metre guesses:
# metres = exp(BLEND_W*log(size) + (1-BLEND_W)*log(depth)). 0.5 = geometric mean until 10+
# measured `distance/` photos exist; `check_distances` reports the best weight it finds.
BLEND_W = 0.5
MIN_PHOTOS_FOR_W = 10
DISAGREE_RATIO = 1.5  # metres estimates further apart than this factor are flagged
EXIF_IFD = 0x8769
TAG_FOCAL_35MM = 0xA405


# ------------------------------------------------------------ opening photos

def _register_heif():
    """Let PIL open iPhone .heic/.heif photos (needs `pip install pillow-heif`)."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return False
    register_heif_opener()
    return True


def open_photo(path):
    """Open an uploaded photo: (upright RGB image, focal_px or None).

    Works for JPG, PNG, WebP and iPhone HEIC. Raises ValueError with a plain message
    when the file is not a photo PIL can read.
    """
    heif = _register_heif()
    name = Path(path).name
    try:
        image = Image.open(path)
        image.load()
    except Exception as e:
        hint = ("iPhone HEIC photos need pillow-heif: run `!pip install pillow-heif` and "
                "re-run this cell, or export the photo as JPG."
                if Path(path).suffix.lower() in (".heic", ".heif") and not heif else
                "Use a JPG, PNG or HEIC photo.")
        raise ValueError(f"Could not open {name} as a photo. {hint}") from e
    focal = focal_px_from_exif(image)  # before rotating: 36 mm is the sensor's long side
    return ImageOps.exif_transpose(image).convert("RGB"), focal


# ------------------------------------------------------------ size prior

def focal_px_from_exif(image):
    """Focal length in pixels from EXIF FocalLengthIn35mmFilm, or None if missing.

    35mm-equivalent focal length is defined against a 36 mm wide frame, so
    focal_px = f35 / 36 * image width (in pixels).
    """
    try:
        f35 = image.getexif().get_ifd(EXIF_IFD).get(TAG_FOCAL_35MM)
    except Exception:
        return None
    if not f35:
        return None
    return float(f35) / 36.0 * image.size[0]


def size_prior_metres(label, box, image_size, focal_px):
    """Pinhole estimate from the box's longest side. box = (x0,y0,x1,y1) fractions."""
    if not focal_px or label not in REAL_SIZE_M:
        return None
    w, h = image_size
    side = max((box[2] - box[0]) * w, (box[3] - box[1]) * h)
    return focal_px * REAL_SIZE_M[label] / side if side > 0 else None


# ------------------------------------------------------------ depth models

def depth_models(relative=RELATIVE_MODEL, metric=METRIC_MODEL, device=None):
    """Load both depth models (Colab only).

    Returns (relative_fn, metric_fn); each maps a PIL image to a float HxW numpy array at the
    image's own size. relative: bigger = nearer (inverse depth, arbitrary scale);
    metric: metres.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def make(name):
        processor = AutoImageProcessor.from_pretrained(name)
        model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

        def run(image):
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs).predicted_depth
            out = torch.nn.functional.interpolate(out[:, None], size=image.size[::-1],
                                                  mode="bicubic", align_corners=False)
            return out[0, 0].float().cpu().numpy()
        return run
    return make(relative), make(metric)


def blend_metres(depth, size, w=None):
    """Log-space blend of the depth-model and size-prior metres (w = weight of size).
    Either may be None: the other is returned. w=0.5 is the geometric mean."""
    w = BLEND_W if w is None else w
    if depth and size:
        return float(np.exp(w * np.log(size) + (1 - w) * np.log(depth)))
    return depth or size or None


def box_value(depth_map, box, inner=0.5):
    """Median depth in the middle `inner` share of the box (edges often hit the background)."""
    h, w = depth_map.shape
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * inner / 2, (box[3] - box[1]) * inner / 2
    x0, x1 = int(max(0, (cx - hw) * w)), int(min(w, np.ceil((cx + hw) * w)))
    y0, y1 = int(max(0, (cy - hh) * h)), int(min(h, np.ceil((cy + hh) * h)))
    patch = depth_map[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    return float(np.median(patch))


# ------------------------------------------------------------ per photo

def estimate_distances(image, detections, relative_fn, metric_fn, focal_px=None):
    """Add distances to detections ([{"label","score","box"}]) and order them near -> far.

    Each result gets: `relative` (depth-model score, bigger = nearer), `metres_depth`,
    `metres_size` (None without EXIF), `metres` (the number to show: log-space blend of the two,
    see BLEND_W, otherwise whichever exists), `disagree` (True if the two differ by >1.5x),
    `confidence` ("low" for keys or disagreeing estimates), `rank` (1 = nearest).
    """
    if not detections:
        return []
    rel_map, met_map = relative_fn(image), metric_fn(image)
    out = []
    for d in detections:
        r = dict(d)
        r["relative"] = box_value(rel_map, d["box"])
        r["metres_depth"] = box_value(met_map, d["box"])
        r["metres_size"] = size_prior_metres(d["label"], d["box"], image.size, focal_px)
        both = [m for m in (r["metres_depth"], r["metres_size"]) if m]
        r["metres"] = blend_metres(r["metres_depth"], r["metres_size"])
        r["calibrated_by_photo"] = False
        r["disagree"] = (len(both) == 2 and max(both) / min(both) > DISAGREE_RATIO)
        r["confidence"] = "low" if d["label"] in LOW_CONFIDENCE or r["disagree"] else "ok"
        out.append(r)
    # One photo with 2+ objects that have a size prior (not keys): fit one scale for the metric
    # model from them and apply it to every object in the photo (keys get a calibrated number).
    ratios = [r["metres_size"] / r["metres_depth"] for r in out
              if r["metres_size"] and r["metres_depth"] and r["label"] not in LOW_CONFIDENCE]
    if len(ratios) >= 2:
        scale = statistics.median(ratios)
        for r in out:
            r["metres_depth_raw"] = r["metres_depth"]
            r["metres_depth"] = r["metres_depth"] * scale
            r["metres"] = blend_metres(r["metres_depth"], r["metres_size"])
            r["calibrated_by_photo"] = True
    # order by the relative model (reliable); metres only break ties
    for rank, r in enumerate(sorted(out, key=lambda r: -r["relative"]), 1):
        r["rank"] = rank
    return sorted(out, key=lambda r: r["rank"])


def describe(results):
    """One plain line per object, nearest first."""
    lines = []
    for r in results:
        m = "distance unknown" if r["metres"] is None else f"about {r['metres']:.1f} m"
        note = " (the two estimates disagree)" if r["disagree"] else ""
        note += " (rough: keys are small)" if r["label"] in LOW_CONFIDENCE else ""
        lines.append(f"{r['rank']}. {r['label']} {r['score'] * 100:.0f}% sure: {m}{note}")
    return lines


def draw_results(image, results):
    """Copy of the image with a box and "rank. label, metres" on every object."""
    from PIL import ImageDraw
    from .autobox import COLOURS
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for r in results:
        b = r["box"]
        xy = (b[0] * w, b[1] * h, b[2] * w, b[3] * h)
        colour = COLOURS.get(r["label"], "#ff7f0e")
        d.rectangle(xy, outline=colour, width=3)
        m = "?" if r["metres"] is None else f"~{r['metres']:.1f} m"
        d.text((xy[0] + 4, xy[1] + 2), f"{r['rank']}. {r['label']} {m}", fill=colour)
    return out


# --------------------------------------------- check against measured photos

def _median_ratio(pairs):
    ratios = [t / p for t, p in pairs if p and p > 0]
    return statistics.median(ratios) if ratios else 1.0


def _loo_correct(items, key):
    """Predictions scaled by the median true/pred ratio of all OTHER photos (leave-one-out)."""
    out = []
    for i, it in enumerate(items):
        others = [(o["true"], o[key]) for j, o in enumerate(items) if j != i and o.get(key)]
        out.append(it[key] * _median_ratio(others) if it.get(key) and others else None)
    return out


def check_distances(photos, detect_fn, relative_fn, metric_fn, min_score=0.3):
    """Score the estimates on tape-measured photos.

    photos: [(path, object, true_metres)] (see `load_distance_photos`);
    detect_fn(image) -> detections. Uses the best-scoring box of the photographed object.
    Returns {"rows": [...], "summary": {...}, "missed": [names]}.
    """
    rows, missed = [], []
    for path, obj, true in photos:
        image, focal = open_photo(path)
        dets = [d for d in detect_fn(image) if d["label"] == obj and d["score"] >= min_score]
        if not dets:
            missed.append(str(path))
            continue
        best = max(dets, key=lambda d: d["score"])
        r = estimate_distances(image, [best], relative_fn, metric_fn, focal)[0]
        rows.append({"photo": getattr(path, "name", str(path)), "object": obj, "true": true,
                     "depth": r["metres_depth"], "size": r["metres_size"],
                     "shown": r["metres"], "relative": r["relative"]})
    summary = {"photos": len(photos), "found": len(rows)}
    for key in ("depth", "size", "shown", "geo"):
        have = [r for r in rows if r[key]]
        if not have:
            continue
        errs = [abs(r[key] - r["true"]) for r in have]
        rel = [abs(r[key] - r["true"]) / r["true"] for r in have]
        summary[key] = {"n": len(have), "median_error_m": statistics.median(errs),
                        "median_error_pct": 100 * statistics.median(rel)}
        if len(have) > 2:  # needs other photos to fit a scale on
            corrected = _loo_correct(have, key)
            rel_c = [abs(c - r["true"]) / r["true"] for c, r in zip(corrected, have) if c]
            summary[key]["median_error_pct_after_scale_fix"] = 100 * statistics.median(rel_c)
    for r in rows:  # the combined guesses, scored like the others
        r["geo"] = blend_metres(r["depth"], r["size"], 0.5)
    both = [r for r in rows if r["depth"] and r["size"]]
    if both:
        w, w_err = fit_blend_weight(both)
        summary["blend_w"] = {"best_w": w, "n": len(both), "median_error_pct_loo": w_err,
                              "enough_photos": len(both) >= MIN_PHOTOS_FOR_W}
        for r in both:
            r["blend"] = blend_metres(r["depth"], r["size"], w)
    summary["order"] = _order_accuracy(rows)
    return {"rows": rows, "summary": summary, "missed": missed}


def fit_blend_weight(rows, grid=None):
    """Best size-weight w (0..1, step 0.1) for the log blend, chosen leave-one-out: each photo
    is scored with the w that is best on all the OTHER photos. Returns (w fitted on all,
    median % error of the leave-one-out scores)."""
    grid = [i / 10 for i in range(11)] if grid is None else grid

    def err(r, w):
        return abs(blend_metres(r["depth"], r["size"], w) - r["true"]) / r["true"]

    def best(sub):
        # ties go to the grid value closest to 0.5
        return min(grid, key=lambda w: (statistics.median(err(r, w) for r in sub), abs(w - 0.5)))
    loo = [err(r, best(rows[:i] + rows[i + 1:]) if len(rows) > 1 else 0.5)
           for i, r in enumerate(rows)]
    return best(rows), 100 * statistics.median(loo)


def _order_accuracy(rows):
    """Of all pairs of photos of the same object at different measured distances, how often
    does each estimate put the nearer one nearer? (Relative scores are only comparable inside
    one photo, so they are not scored here.)"""
    out = {}
    for key in ("depth", "size"):
        ok = total = 0
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a["object"] != b["object"] or a["true"] == b["true"] or not a[key] or not b[key]:
                    continue
                total += 1
                ok += (a[key] < b[key]) == (a["true"] < b["true"])
        if total:
            out[key] = {"pairs": total, "correct": ok}
    return out


def format_check(result):
    s = result["summary"]
    lines = [f"Distance check: the detector found the object in {s['found']} of "
             f"{s['photos']} measured photos."]
    names = {"depth": "depth model", "size": "object-size estimate", "shown": "number shown (mix)",
             "geo": "geometric mean of the two"}
    for key, name in names.items():
        if key not in s:
            continue
        e = s[key]
        line = (f"- {name}: typically off by {e['median_error_m']:.2f} m "
                f"({e['median_error_pct']:.0f}% of the real distance), on {e['n']} photos")
        if "median_error_pct_after_scale_fix" in e:
            line += f"; {e['median_error_pct_after_scale_fix']:.0f}% after a leave-one-out scale fix"
        lines.append(line)
    if "blend_w" in s:
        b = s["blend_w"]
        line = (f"- best blend: {b['best_w']:.1f} of the size estimate + {1 - b['best_w']:.1f} of the "
                f"depth model (typically {b['median_error_pct_loo']:.0f}% off, leave-one-out, "
                f"{b['n']} photos)")
        if not b["enough_photos"]:
            line += f"; fewer than {MIN_PHOTOS_FOR_W} photos, so too noisy to trust: keep BLEND_W = 0.5"
        lines.append(line)
    for key, o in s["order"].items():
        lines.append(f"- near/far order from the {key} estimate: right in {o['correct']} of "
                     f"{o['pairs']} photo pairs")
    if result["missed"]:
        lines.append(f"- detector missed the object in {len(result['missed'])} photos "
                     "(not scored; these count against the detector, not distance)")
    lines.append("Near/far order is reliable; metres are approximate.")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--root", default="data/detection")
    c.add_argument("--checkpoint", default="detector.pt")
    a = p.parse_args(argv)
    from .detection_data import load_distance_photos
    from .detector import detect, load_detector
    photos = load_distance_photos(a.root)
    if not photos:
        print("No measured photos found (claudeknows_photos/distance/, named like keys_1.0m.jpg).")
        return
    model, classes = load_detector(a.checkpoint)
    rel, met = depth_models()
    result = check_distances(photos, lambda im: detect(model, im, classes), rel, met)
    print(format_check(result))


if __name__ == "__main__":
    main()
