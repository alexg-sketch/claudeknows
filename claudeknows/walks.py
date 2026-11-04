"""Phase 2, step 1: import "walk" videos (Alex walking round the house).

New videos in `<root>/walks/` become sharp, non-repeating frames in
`<out>/<video name>/`. A manifest (`manifest.json`) remembers every video that
was processed (by a fingerprint of its content, so renaming does not matter),
so running this again only handles new videos. Nothing here is ever committed:
walk footage is Alex's private data.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .frames import MAX_SIDE, VIDEO_EXTS, extract_frames

MANIFEST = "manifest.json"
HASH_SIZE = 8  # 8x8 difference hash = 64 bits


def dhash(bgr, size=HASH_SIZE):
    """Perceptual 'difference hash': 64-bit int; similar pictures differ in few bits."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def drop_near_duplicates(paths, max_distance=4):
    """Keep a frame only if it differs from the last KEPT frame by more than
    `max_distance` bits (standing still gives many near-identical frames).
    Deletes the dropped files. Returns (kept_paths, n_dropped)."""
    kept, last, dropped = [], None, 0
    for p in sorted(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h = dhash(img)
        if last is not None and hamming(h, last) <= max_distance:
            Path(p).unlink()
            dropped += 1
            continue
        kept.append(Path(p))
        last = h
    return kept, dropped


def fingerprint(path, chunk=1 << 20):
    """Cheap content fingerprint: size + hash of first and last MB."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:24]


def load_manifest(out_dir):
    p = Path(out_dir) / MANIFEST
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"videos": {}}


def import_walks(walks_dir, out_dir, fps=2.0, max_distance=4, backend="auto",
                 max_side=MAX_SIDE):
    """Process every NEW video in `walks_dir`. Returns a list of per-video dicts
    (name, kept, blurry, duplicates). Missing/empty folder -> empty list."""
    walks_dir, out_dir = Path(walks_dir), Path(out_dir)
    if not walks_dir.is_dir():
        return []
    videos = sorted(p for p in walks_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    manifest = load_manifest(out_dir)
    results = []
    for v in videos:
        fp = fingerprint(v)
        if fp in manifest["videos"]:
            continue
        frames_dir = out_dir / v.stem
        _, blurry = extract_frames(v, frames_dir, fps=fps, backend=backend,
                                   max_side=max_side)
        kept, dups = drop_near_duplicates(frames_dir.glob("*.jpg"), max_distance)
        info = {"name": v.name, "kept": len(kept), "blurry": blurry,
                "duplicates": dups}
        manifest["videos"][fp] = {**info, "frames_dir": v.stem}
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / (MANIFEST + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=1))
        tmp.replace(out_dir / MANIFEST)  # safe against a Colab disconnect
        results.append(info)
    return results


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Import new walk videos into frames")
    ap.add_argument("walks_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-distance", type=int, default=4)
    a = ap.parse_args(argv)
    res = import_walks(a.walks_dir, a.out_dir, a.fps, a.max_distance)
    if not res:
        print("No new walk videos.")
    for r in res:
        print(f"{r['name']}: kept {r['kept']} frames "
              f"({r['blurry']} blurry, {r['duplicates']} near-duplicates dropped)")


if __name__ == "__main__":
    main()
