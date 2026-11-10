import cv2
import numpy as np

from claudeknows.walks import dhash, drop_near_duplicates, hamming, import_walks


def make_video(path, n_frames=40, fps=10, size=(96, 64)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()
    return path


def test_dhash_similar_vs_different():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + 2, 0, 255).astype(np.uint8)
    c = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert hamming(dhash(a), dhash(b)) <= 4
    assert hamming(dhash(a), dhash(c)) > 10


def test_drop_near_duplicates(tmp_path):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    c = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for name, img in [("1", a), ("2", a), ("3", c)]:
        cv2.imwrite(str(tmp_path / f"{name}.png"), img)
    kept, dropped = drop_near_duplicates(tmp_path.glob("*.png"))
    assert dropped == 1 and len(kept) == 2


def test_import_walks_once_and_missing(tmp_path):
    assert import_walks(tmp_path / "nope", tmp_path / "out") == []
    walks = tmp_path / "walks"
    walks.mkdir()
    make_video(walks / "w1.mp4", n_frames=30, fps=10)
    out = tmp_path / "out"
    first = import_walks(walks, out, backend="opencv")
    assert len(first) == 1 and first[0]["kept"] > 0
    assert list((out / "w1").glob("*.jpg"))
    # second run: nothing new, even if renamed
    (walks / "w1.mp4").rename(walks / "renamed.mp4")
    assert import_walks(walks, out, backend="opencv") == []
    make_video(walks / "w2.mp4", n_frames=20, fps=10)
    assert [r["name"] for r in import_walks(walks, out, backend="opencv")] == ["w2.mp4"]
