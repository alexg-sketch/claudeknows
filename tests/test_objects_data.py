import io
import subprocess
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from claudeknows import frames, openimages
from claudeknows.objects import (OBJECT_NAMES, ObjectDataset, import_own, scan,
                                 split_by_group)

ROOT = Path(__file__).resolve().parents[1]


def make_video(path, n_frames=40, fps=10, sharp=True, size=(96, 64)):
    """Tiny synthetic video (sharp noise = sharp frames, flat = blurry)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, fps, size)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        if sharp:
            f = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        else:
            f = np.full((size[1], size[0], 3), 120, dtype=np.uint8)
        w.write(f)
    w.release()
    return path


def make_photo(path, color=(200, 30, 30), size=(80, 60)):
    Image.new("RGB", size, color).save(path)


def test_blur_score_separates_sharp_from_flat():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    flat = np.full((64, 64, 3), 100, dtype=np.uint8)
    assert frames.blur_score(sharp) > frames.BLUR_THRESHOLD > frames.blur_score(flat)


@pytest.mark.parametrize("backend", ["opencv", "ffmpeg"])
def test_extract_frames_about_two_per_second(tmp_path, backend):
    if backend == "ffmpeg":
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("ffmpeg not installed")
    v = make_video(tmp_path / "clip.mp4", n_frames=40, fps=10)  # 4 seconds
    kept, skipped = frames.extract_frames(v, tmp_path / "out", fps=2, backend=backend)
    assert 7 <= kept <= 9 and skipped == 0
    files = sorted((tmp_path / "out").glob("clip__f*.jpg"))
    assert len(files) == kept


def test_blurry_video_keeps_one_frame_rather_than_nothing(tmp_path):
    v = make_video(tmp_path / "flat.mp4", sharp=False)
    kept, skipped = frames.extract_frames(v, tmp_path / "out", backend="opencv")
    assert kept == 1 and skipped >= 1


def test_blurry_frames_are_skipped_among_sharp_ones(tmp_path):
    # first half noisy (sharp), second half flat (blurry)
    path = tmp_path / "mix.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    rng = np.random.default_rng(0)
    for i in range(40):
        f = (rng.integers(0, 255, (64, 96, 3), dtype=np.uint8) if i < 20
             else np.full((64, 96, 3), 90, dtype=np.uint8))
        w.write(f)
    w.release()
    kept, skipped = frames.extract_frames(path, tmp_path / "out", backend="opencv")
    assert kept >= 3 and skipped >= 3


def test_import_own_handles_photos_videos_and_missing_folders(tmp_path):
    drive = tmp_path / "claudeknows_photos"
    (drive / "keys").mkdir(parents=True)
    make_photo(drive / "keys" / "IMG_1.JPG")
    make_video(drive / "keys" / "a.mp4")
    (drive / "keys" / "notes.txt").write_text("not media")
    (drive / "keys" / "broken.mov").write_bytes(b"not a video")
    (drive / "shoes").mkdir()  # empty folder; water_bottle folder missing
    root = tmp_path / "objects"
    logs = []
    summary = import_own(drive, root, backend="opencv", log=logs.append)
    assert summary["keys"]["photos"] == 1 and summary["keys"]["videos"] == 1
    assert summary["keys"]["unreadable"] == ["broken.mov"]
    assert summary["shoes"]["videos"] == 0 and summary["water_bottle"]["photos"] == 0
    assert (root / "own" / "keys" / "photo_IMG_1.jpg").exists()
    assert list((root / "own" / "keys").glob("vid_a__f*.jpg"))
    # re-running does not duplicate work
    again = import_own(drive, root, backend="opencv", log=logs.append)
    assert again["keys"]["frames"] == 0 and again["keys"]["videos"] == 1


def test_import_own_with_no_drive_folder_at_all(tmp_path):
    summary = import_own(tmp_path / "nope", tmp_path / "objects", log=lambda *_: None)
    assert all(v["photos"] == v["videos"] == 0 for v in summary.values())
    classes, samples = scan(tmp_path / "objects")
    assert classes == OBJECT_NAMES and samples == []


def _tree(root, spec):
    """spec: {(source, cls): {group: n_frames}}"""
    for (source, cls), groups in spec.items():
        d = root / source / cls
        d.mkdir(parents=True, exist_ok=True)
        for g, n in groups.items():
            for k in range(n):
                name = f"{g}__f{k:04d}.jpg" if g.startswith("vid_") else f"{g}.jpg"
                make_photo(d / name)
    return root


def test_split_is_by_video_never_by_frame(tmp_path):
    root = _tree(tmp_path, {
        ("own", "keys"): {f"vid_v{i}": 6 for i in range(5)},
        ("web", "keys"): {f"id{i}": 1 for i in range(10)},
        ("own", "shoes"): {"vid_only": 8},  # a single video cannot be split
    })
    classes, samples = scan(root)
    train, test = split_by_group(samples)
    assert len(train) + len(test) == len(samples)
    assert {s.group for s in train}.isdisjoint({s.group for s in test})
    own_keys_test = {s.group for s in test if s.source == "own" and s.label == 1}
    assert len(own_keys_test) == 1  # whole video held out
    assert sum(1 for s in test if s.group in own_keys_test) == 6
    assert {s.group for s in train if s.label == 2} == {"vid_only"}
    assert not [s for s in test if s.label == 2]


def test_split_is_deterministic(tmp_path):
    root = _tree(tmp_path, {("web", "keys"): {f"id{i}": 1 for i in range(20)}})
    _, samples = scan(root)
    assert split_by_group(samples, seed=3)[1] == split_by_group(samples, seed=3)[1]


def test_object_dataset_yields_image_and_label(tmp_path):
    root = _tree(tmp_path, {("web", "shoes"): {"a": 1}})
    _, samples = scan(root)
    img, label = ObjectDataset(samples)[0]
    assert img.size == (80, 60) and label == 2


# ---- Open Images downloader (offline: file:// URLs stand in for the real hosts) ----

def _jpeg(size=(200, 100), color=(10, 200, 10)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def test_crop_to_box_and_too_small_is_rejected():
    img = Image.open(io.BytesIO(_jpeg((400, 200))))
    out = openimages.crop_and_shrink(img, (0.25, 0.75, 0.0, 1.0))
    assert out is not None and 200 <= out.size[0] <= 256 and out.size[1] <= 256
    assert openimages.crop_and_shrink(img, (0.0, 0.05, 0.0, 0.05)) is None


def test_find_candidates_and_download_with_404s(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "ImageID,Source,LabelName,Confidence\n"
        "aaa,verification,/m/0118n_9r,1.0\n"   # bottle photo, has a box
        "bbb,verification,/m/0118n_9r,1.0\n"   # bottle photo, no box -> dropped
        "ccc,verification,/m/0118n_9r,0.0\n"   # negative label
        "ddd,verification,/m/03v3yw,1.0\n"     # key
        "eee,verification,/m/03lnq3,1.0\n"     # keychain (will 404)
    )
    head = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
    boxes = tmp_path / "boxes.csv"
    boxes.write_text(
        head
        + "aaa,xclick,/m/04dr76w,1,0.1,0.5,0.1,0.9,0,0,0,0,0\n"
        + "aaa,xclick,/m/04dr76w,1,0.2,0.4,0.2,0.3,0,0,0,0,0\n"  # smaller box
        + "ccc,xclick,/m/04dr76w,1,0.1,0.5,0.1,0.9,0,0,0,0,0\n"
        + "s1,xclick,/m/09j5n,1,0.0,1.0,0.0,1.0,0,0,0,0,0\n"
        + "s2,xclick,/m/09j5n,1,0.0,1.0,0.0,1.0,0,0,1,0,0\n"  # group of shoes -> dropped
    )
    cands = openimages.find_candidates(
        10, labels_url=labels.as_uri(), boxes_url=boxes.as_uri(), log=lambda *_: None)
    assert list(cands["water_bottle"]) == ["aaa"]
    assert cands["water_bottle"]["aaa"] == (0.1, 0.5, 0.1, 0.9)  # the biggest box
    assert sorted(cands["keys"]) == ["ddd", "eee"] and cands["keys"]["ddd"] is None
    assert list(cands["shoes"]) == ["s1"]

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name in ("aaa", "ddd", "s1"):
        (imgs / f"{name}.jpg").write_bytes(_jpeg())
    url = (imgs.as_uri() + "/{}.jpg")
    out = tmp_path / "out"
    saved = openimages.download_object("keys", cands["keys"], out, cap=5,
                                       image_url=url, log=lambda *_: None)
    assert saved == ["ddd"]  # eee is missing -> skipped, no crash
    assert [p.name for p in out.glob("*.jpg")] == ["ddd.jpg"]
    capped = openimages.download_object("shoes", {"s1": None, "aaa": None}, tmp_path / "o2",
                                        cap=1, image_url=url, log=lambda *_: None)
    assert len(capped) == 1


def test_credits_file(tmp_path):
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title\n"
        "ddd,train,u,https://flickr/x,https://cc/by/2.0/,https://flickr/p/,Jane Doe,My keys\n")
    path = openimages.write_credits(tmp_path, {"keys": ["ddd", "zzz"]}, meta_url=meta.as_uri())
    text = path.read_text()
    assert "Jane Doe" in text and "zzz" in text


# ---- the repo is public: personal footage must never be committable ----

def test_gitignore_blocks_personal_footage():
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in ("claudeknows_photos/", "data/", "*.mov", "*.MOV", "*.mp4", "*.heic"):
        assert pattern in ignore.split()
