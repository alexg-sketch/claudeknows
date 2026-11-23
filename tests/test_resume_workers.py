import random

import pytest
import torch
from torch.utils.data import DataLoader

from claudeknows import detector
from tests.test_zoom_ema import _samples

KW = dict(batch_size=2, pretrained=False, max_batches=2, device="cpu", warmup=1,
          zoom_crop_p=0.5, ema_decay=0.9)


def _gap(a, b):
    return max(float((v - b.state_dict()[k]).abs().max())
               for k, v in a.state_dict().items() if v.dtype.is_floating_point)


class _Stop(Exception):
    pass


def _stop_before(epoch_line):
    def log(msg):
        if str(msg).startswith(epoch_line):
            raise _Stop
    return log


def test_resume_gives_same_weights_as_unbroken_run(tmp_path):
    samples = _samples(tmp_path)
    straight, _ = detector.train_detector(samples, epochs=4, log=lambda *_: None, **KW)
    path = str(tmp_path / "resume.pt")
    with pytest.raises(_Stop):  # cut off right after epoch 2 was saved
        detector.train_detector(samples, epochs=4, resume_path=path, log=_stop_before("epoch 3/4"), **KW)
    msgs = []
    resumed, hist = detector.train_detector(samples, epochs=4, resume_path=path, log=msgs.append, **KW)
    assert any("resuming" in m and "2 of 4" in m for m in msgs)
    assert len(hist) == 4
    assert _gap(straight, resumed) < 1e-5
    assert not (tmp_path / "resume.pt").exists()  # removed after a normal finish


def test_keep_resume_survives_a_crash_after_training(tmp_path):
    """The CLI deletes the resume file only after saving, so a crash between loses nothing."""
    samples = _samples(tmp_path)
    path = str(tmp_path / "resume.pt")
    first, _ = detector.train_detector(samples, epochs=2, resume_path=path, keep_resume=True,
                                       log=lambda *_: None, **KW)
    assert (tmp_path / "resume.pt").exists()
    msgs = []  # e.g. the cut-off step crashed: the rerun picks the finished model straight back up
    again, hist = detector.train_detector(samples, epochs=2, resume_path=path, log=msgs.append, **KW)
    assert any("2 of 2" in m for m in msgs) and len(hist) == 2
    assert _gap(first, again) < 1e-6


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04 cut off"])
def test_broken_resume_file_is_set_aside_and_training_starts_fresh(tmp_path, content):
    samples = _samples(tmp_path)
    path = tmp_path / "resume.pt"
    path.write_bytes(content)  # what Drive left after an interrupted save
    msgs = []
    _, hist = detector.train_detector(samples, epochs=1, resume_path=str(path), log=msgs.append, **KW)
    assert any("could not be read" in m for m in msgs)
    assert len(hist) == 1
    assert (tmp_path / "resume.pt.broken").read_bytes() == content


def test_resume_with_different_settings_starts_fresh(tmp_path):
    samples = _samples(tmp_path)
    path = str(tmp_path / "resume.pt")
    with pytest.raises(_Stop):
        detector.train_detector(samples, epochs=3, resume_path=path, log=_stop_before("epoch 2/3"), **KW)
    msgs = []
    _, hist = detector.train_detector(samples, epochs=2, resume_path=path, log=msgs.append, **KW)
    assert any("different settings" in m for m in msgs)
    assert len(hist) == 2


class _Draws(torch.utils.data.Dataset):
    def __init__(self):
        self.rng = random.Random(1)  # copied into every worker, like the transforms' generators
        self.transform = self
        self.transform.reseed = lambda v: self.rng.seed(v)

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.tensor(self.rng.random())


def test_workers_do_not_repeat_the_same_draws():
    ds = _Draws()
    loader = DataLoader(ds, batch_size=4, num_workers=2, worker_init_fn=detector._seed_worker)
    first, second = [b for b in loader]
    assert not torch.equal(first, second)  # the two workers drew different numbers


def test_training_with_two_workers_finishes(tmp_path):
    samples = _samples(tmp_path)
    _, hist = detector.train_detector(samples, epochs=1, workers=2, log=lambda *_: None, **KW)
    assert len(hist) == 1
