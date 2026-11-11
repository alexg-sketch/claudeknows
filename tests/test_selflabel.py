import json

import numpy as np
import torch
from PIL import Image

from claudeknows import selflabel as sl
from claudeknows.detection_data import DetSample

BOTTLE = (0.2, 0.2, 0.5, 0.8)
SHOE = (0.6, 0.6, 0.9, 0.9)


def cut(_label):
    return 0.5


def make_walk(tmp_path, n=5, name="walk1"):
    d = tmp_path / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)).save(d / f"f{i:03d}.jpg")
    return d


def scripted(per_frame):
    """Fake labeller: returns per_frame[k] on its k-th call."""
    calls = iter(per_frame)

    def f(*_args):
        return next(calls)
    return f


def test_agree_matches_same_label_only():
    det = [{"label": "water_bottle", "score": 0.9, "box": BOTTLE},
           {"label": "shoes", "score": 0.8, "box": SHOE},
           {"label": "keys", "score": 0.2, "box": SHOE}]  # below cut-off: ignored
    owl = [("water_bottle", 0.7, (0.21, 0.2, 0.5, 0.8)), ("keys", 0.6, SHOE)]
    agreed, single = sl.agree(det, owl, cut, 0.3)
    assert [a["label"] for a in agreed] == ["water_bottle"]
    assert abs(agreed[0]["score"] - 0.7) < 1e-9
    assert sorted((s["label"], s["seen_by"]) for s in single) == [("keys", "owlv2"),
                                                                  ("shoes", "detector")]


def test_steady_keeps_steady_box_drops_flicker():
    """Fake 5-frame walk: a bottle in every frame, a shoe in one frame only."""
    frames = [[{"label": "water_bottle", "box": BOTTLE}] for _ in range(5)]
    frames[2].append({"label": "shoes", "box": SHOE})
    flags = sl.steady(frames)
    assert all(f[0] for f in flags)
    assert flags[2] == [True, False]


def test_label_walk_rules_and_answers(tmp_path):
    d = make_walk(tmp_path)
    bottle_d = {"label": "water_bottle", "score": 0.9, "box": BOTTLE}
    bottle_o = ("water_bottle", 0.8, BOTTLE)
    shoe_d = {"label": "shoes", "score": 0.9, "box": SHOE}
    det = [[bottle_d], [bottle_d], [bottle_d, shoe_d], [bottle_d],
           [bottle_d, {"label": "keys", "score": 0.1, "box": SHOE}]]  # low-score keys: ignored
    owl = [[bottle_o]] * 5
    stats = sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, log=lambda *_: None)
    assert stats["agreed"] == 5 and stats["single_asked"] == 1
    labels = sl.load_labels(d)
    # frame with one agreed box + one detector-only confident box is held back
    assert labels["f002.jpg"]["status"] == sl.HELD
    # a frame with only a low-score disagreement is kept
    assert labels["f004.jpg"]["status"] == sl.TRAIN
    qs = sl.load_questions(d)
    assert len(qs) == 1 and qs[0]["frame"] == "f002.jpg" and qs[0]["label"] == "shoes"
    assert len(sl.self_samples(tmp_path)) == 4
    # after "yes" the frame is used, with both boxes
    sl.apply_answers(d, {qs[0]["id"]: True})
    labels = sl.load_labels(d)
    assert labels["f002.jpg"]["status"] == sl.TRAIN
    assert sorted(b["label"] for b in labels["f002.jpg"]["boxes"]) == ["shoes", "water_bottle"]
    samples = sl.self_samples(tmp_path)
    assert len(samples) == 5 and all(s.by == sl.SELF and s.source == "walk" for s in samples)


def test_answer_no_frees_frame_without_box(tmp_path):
    d = make_walk(tmp_path, n=3)
    det = [[], [], []]
    owl = [[], [("keys", 0.9, SHOE)], []]
    sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, log=lambda *_: None)
    q = sl.load_questions(d)[0]
    assert sl.load_labels(d)["f001.jpg"]["status"] == sl.HELD
    counts = sl.apply_answers(d, {q["id"]: False})
    e = sl.load_labels(d)["f001.jpg"]
    assert e["boxes"] == [] and e["status"] == sl.EMPTY
    assert counts[sl.EMPTY] == 3


def test_unsteady_low_score_dropped_and_frame_held(tmp_path):
    d = make_walk(tmp_path, n=5)
    det = [[] for _ in range(5)]
    det[2] = [{"label": "shoes", "score": 0.55, "box": SHOE}]
    owl = [[] for _ in range(5)]
    owl[2] = [("shoes", 0.4, SHOE)]
    stats = sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, log=lambda *_: None)
    assert stats["unsteady_dropped"] == 1 and sl.load_questions(d) == []
    assert sl.load_labels(d)["f002.jpg"]["status"] == sl.HELD


def test_questions_sorted_by_disagreement_and_capped(tmp_path):
    d = make_walk(tmp_path, n=4)
    det = [[] for _ in range(4)]
    owl = [[("keys", 0.9, (0.1 * i, 0.1, 0.1 * i + 0.2, 0.3))] for i in range(4)]
    loss = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3}

    def loss_fn(_im, boxes):
        return loss[round(boxes[-1]["box"][0] * 10)]
    stats = sl.label_walk(d, scripted(det), scripted(owl), cut, 0.3, loss_fn=loss_fn,
                          max_questions=3, log=lambda *_: None)
    qs = sl.load_questions(d)
    assert [q["disagreement"] for q in qs] == [0.9, 0.5, 0.3]
    assert stats["questions_over_cap"] == 1
    # the frame whose question did not make the cut stays held back
    assert sl.load_labels(d)["f000.jpg"]["status"] == sl.HELD


def _tiny_rcnn():
    """Small Faster R-CNN with a plain conv backbone. (An untrained MobileNet backbone
    gives near-constant features, so no box could differ from another.)"""
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    bb = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
                             torch.nn.Conv2d(16, 16, 3, padding=1, stride=2), torch.nn.ReLU())
    bb.out_channels = 16
    return FasterRCNN(bb, num_classes=4, min_size=96, max_size=128,
                      rpn_anchor_generator=AnchorGenerator(((32,),), ((1.0,),)),
                      box_roi_pool=MultiScaleRoIAlign(["0"], 7, 2), representation_size=64)


def test_review_loss_higher_for_wrong_box():
    torch.manual_seed(0)
    model = _tiny_rcnn()
    img = np.full((96, 128, 3), 30, dtype=np.uint8)
    img[20:80, 30:60] = (240, 240, 40)
    im = Image.fromarray(img)
    t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)
    target = {"boxes": torch.tensor([[30.0, 20.0, 60.0, 80.0]]), "labels": torch.tensor([1])}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(40):  # a short fit so the model knows where the bottle is
        loss = sum(model([t], [target]).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    right = [{"label": "water_bottle", "box": (30 / 128, 20 / 96, 60 / 128, 80 / 96)}]
    wrong = [{"label": "water_bottle", "box": (0.75, 0.0, 0.98, 0.3)}]
    assert sl.review_loss(model, im, wrong) > sl.review_loss(model, im, right)
    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_review_loss_runs_on_real_detector():
    from claudeknows.detector import build_detector
    model = build_detector(pretrained=False, min_size=128, max_size=160).eval()
    im = Image.fromarray(np.zeros((96, 128, 3), dtype=np.uint8))
    v = sl.review_loss(model, im, [{"label": "keys", "box": SHOE}, {"label": "shoes", "box": BOTTLE}])
    assert np.isfinite(v) and not model.training


def test_cap_self_share():
    base = [object() for _ in range(70)]
    extra = [object() for _ in range(100)]
    out = sl.cap_self_share(base, extra, 0.3, log=lambda *_: None)
    assert len(out) == 100 and sum(o in extra for o in out) == 30
    assert len(sl.cap_self_share(base, extra[:10], 0.3)) == 80


def test_self_samples_skip_gold(tmp_path):
    d = make_walk(tmp_path, n=2)
    sl.label_walk(d, scripted([[{"label": "keys", "score": 0.9, "box": SHOE}]] * 2),
                  scripted([[("keys", 0.9, SHOE)]] * 2), cut, 0.3, log=lambda *_: None)
    assert len(sl.self_samples(tmp_path)) == 2
    manifest = {"images": [{"source": "walk", "group": "walk1", "sha256": "x"}]}
    assert sl.self_samples(tmp_path, gold=manifest, log=lambda *_: None) == []
    assert isinstance(sl.self_samples(tmp_path)[0], DetSample)
    json.loads((d / sl.LABELS_FILE).read_text())


def test_cli_questions_and_answer(tmp_path, capsys):
    d = make_walk(tmp_path, n=3)
    sl.label_walk(d, scripted([[], [], []]), scripted([[], [("keys", 0.9, SHOE)], []]), cut, 0.3,
                  log=lambda *_: None)
    sl.main(["questions", str(tmp_path)])
    assert "walk1:0" in capsys.readouterr().out
    sl.main(["answer", str(tmp_path), "--yes", "walk1:0"])
    assert sl.load_labels(d)["f001.jpg"]["status"] == sl.TRAIN
    sl.main(["questions", str(tmp_path)])
    assert "walk1:0" not in capsys.readouterr().out
