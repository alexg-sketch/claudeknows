"""Per-object score cut-offs (task 23m)."""
import torch
from PIL import Image

from claudeknows import autobox, detector as D


def _images(label, true_scores=(), false_scores=()):
    """One photo per prediction group: a true box scored s, or a false one."""
    out = []
    for s in true_scores:
        out.append((None, [(label, (0.1, 0.1, 0.5, 0.5))], [(label, s, (0.1, 0.1, 0.5, 0.5))]))
    for s in false_scores:
        out.append((None, [], [(label, s, (0.6, 0.6, 0.9, 0.9))]))
    return out


def test_low_scoring_true_object_gets_low_cutoff_and_high_false_one_high():
    keys = _images("keys", true_scores=[0.3] * 25)                      # real, but low scores
    shoes = _images("shoes", true_scores=[0.9] * 25, false_scores=[0.6] * 25)  # false alarms at 0.6
    t = D.cutoff_table(keys + shoes, ["keys", "shoes"])
    assert t["keys"]["cutoff"] <= 0.3 and t["keys"]["recall"] == 1.0
    assert t["shoes"]["cutoff"] > 0.6 and t["shoes"]["precision"] == 1.0


def test_too_few_validation_boxes_keeps_default():
    t = D.cutoff_table(_images("keys", true_scores=[0.2] * 5), ["keys"])
    assert t["keys"]["cutoff"] == 0.5


def test_detect_uses_saved_cutoffs_and_old_checkpoints_load(tmp_path):
    model = D.build_detector(pretrained=False)
    D.save_detector(model, str(tmp_path / "old.pt"))  # no thresholds saved
    m, classes = D.load_detector(str(tmp_path / "old.pt"))
    assert m.thresholds == {} and D._cut(m, "keys", None) == 0.5
    D.save_detector(model, str(tmp_path / "new.pt"), extra={"thresholds": {"keys": 0.2}})
    m, _ = D.load_detector(str(tmp_path / "new.pt"))
    assert D._cut(m, "keys", None) == 0.2 and D._cut(m, "shoes", None) == 0.5
    assert D._cut(m, "keys", 0.7) == 0.7  # explicit value wins


def test_autobox_per_object_threshold(tmp_path):
    assert autobox.cutoff({"keys": 0.2, "shoes": 0.4}, "keys") == 0.2
    assert autobox.cutoff(0.3, "keys") == 0.3
    assert set(autobox.THRESHOLD) == set(autobox.FILL_THRESHOLD) == set(autobox.OBJECT_NAMES)
    assert all(v == 0.3 for v in autobox.THRESHOLD.values())  # default behaviour unchanged
