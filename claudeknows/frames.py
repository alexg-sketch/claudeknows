"""Turn short phone videos into photos (frames).

~2 frames per second, blurry frames skipped (low variance of the Laplacian).
Uses ffmpeg when it is installed (handles iPhone .MOV / HEVC, and applies the
phone's rotation); otherwise falls back to OpenCV (`opencv-python-headless`).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".hevc")
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic")
BLUR_THRESHOLD = 60.0  # variance of the Laplacian below this = blurry
MAX_SIDE = 256  # frames are saved small; the model only needs ~224 px


def blur_score(bgr):
    """Variance of the Laplacian: higher = sharper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def shrink(bgr, max_side=MAX_SIDE):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return bgr
    return cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _frames_ffmpeg(video, fps, workdir):
    """Yield BGR frames via ffmpeg (auto-rotates, decodes HEVC)."""
    pattern = str(Path(workdir) / "f%05d.jpg")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
           "-vf", f"fps={fps}", "-q:v", "2", pattern]
    subprocess.run(cmd, check=True, capture_output=True)
    for f in sorted(Path(workdir).glob("f*.jpg")):
        img = cv2.imread(str(f))
        if img is not None:
            yield img


def _frames_opencv(video, fps):
    """Yield BGR frames via OpenCV, sampling about `fps` per second."""
    cap = cv2.VideoCapture(str(video))
    try:
        native = cap.get(cv2.CAP_PROP_FPS)
        if not native or native != native or native <= 0:
            native = 30.0
        step = max(1, round(native / fps))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield img
            i += 1
    finally:
        cap.release()


def iter_video_frames(video, fps=2.0, backend="auto"):
    """Yield BGR frames from `video`. backend: auto | ffmpeg | opencv.

    `auto` tries ffmpeg (if installed), then falls back to OpenCV if ffmpeg is
    missing or fails on the file.
    """
    use_ffmpeg = backend == "ffmpeg" or (backend == "auto" and shutil.which("ffmpeg"))
    if use_ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                frames = list(_frames_ffmpeg(video, fps, tmp))
            except (subprocess.CalledProcessError, OSError):
                if backend == "ffmpeg":
                    raise
                frames = []
        if frames or backend == "ffmpeg":
            yield from frames
            return
    yield from _frames_opencv(video, fps)


def extract_frames(video, out_dir, fps=2.0, blur_threshold=BLUR_THRESHOLD,
                   backend="auto", prefix=None):
    """Save the sharp frames of one video into `out_dir`.

    Files are named `<prefix>__f0001.jpg` (prefix defaults to the video's file
    name, so every frame can be traced back to its video). Returns
    (kept, skipped_blurry). If every frame is blurry, the sharpest one is kept
    so a video never silently contributes nothing.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or video.stem
    sharp, best, skipped = [], None, 0
    for img in iter_video_frames(video, fps, backend):
        img = shrink(img)
        score = blur_score(img)
        if score >= blur_threshold:
            sharp.append(img)
        else:
            skipped += 1
            if best is None or score > best[0]:
                best = (score, img)
    if not sharp and best is not None:
        sharp, skipped = [best[1]], skipped - 1
    for n, img in enumerate(sharp, 1):
        cv2.imwrite(str(out_dir / f"{prefix}__f{n:04d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(sharp), skipped
