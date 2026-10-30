from PIL import Image

from claudeknows import detection_data as dd
from claudeknows import detector, openimages


def _folder(root, source, name, files, by, boxes=()):
    d = root / source / name
    d.mkdir(parents=True, exist_ok=True)
    data = {}
    for f, color in files:
        Image.new("RGB", (160, 120), color).save(d / f)
        data[f] = {"by": by, "boxes": [{"label": l, "box": b} for l, b in boxes]}
    dd.save_boxes(d, data)


def test_find_negatives_ranks_best_checked_photos_first(tmp_path):
    L = openimages.NEGATIVE_LABELS  # water bottle, key, keychain, shoe, footwear
    rows = ["ImageID,Source,LabelName,Confidence"]
    rows += [f"four,v,{l},0.0" for l in L[:4]]          # 4 labels checked absent
    rows += [f"two_shoe,v,{l},0.0" for l in (L[0], L[3])]  # 2 absent, incl. shoe
    rows += [f"two_noshoe,v,{l},0.0" for l in (L[0], L[1])]  # 2 absent, no shoe
    rows += [f"one,v,{L[1]},0.0"]                        # 1 absent
    rows += [f"hasshoe,v,{L[0]},0.0", f"hasshoe,v,{L[3]},1.0"]  # a shoe is present
    rows += ["other,v,/m/0abc,0.0"]                      # unrelated label only
    f = tmp_path / "labels.csv"
    f.write_text("\n".join(rows) + "\n")
    got = openimages.find_negatives(100, labels_url=f.as_uri(), log=lambda *_: None)
    assert got == ["four", "two_shoe", "two_noshoe", "one"]
    assert openimages.find_negatives(1, labels_url=f.as_uri(), log=lambda *_: None)[0] == "four"


def test_negatives_are_scanned_separately_and_trained_on(tmp_path):
    root = tmp_path / "det"
    _folder(root, "web", "shoes", [(f"s{i}.jpg", (200, 50, 50)) for i in range(4)],
            "openimages", [("shoes", [0.2, 0.2, 0.7, 0.8])])
    _folder(root, "web", "empty", [(f"n{i}.jpg", (20, 90, 20)) for i in range(3)], "negative")
    pos, _ = dd.scan_detection(root)
    negs = dd.scan_negatives(root)
    assert len(pos) == 4 and len(negs) == 3 and all(n.boxes == () for n in negs)
    # an all-negative batch trains without error
    model, hist = detector.train_detector(negs, epochs=1, batch_size=3, pretrained=False,
                                          max_batches=1, device="cpu", log=lambda *_: None)
    assert len(hist) == 1
    # negatives are capped at ~15% of an epoch: 4 positives -> at most 1 negative
    seen = []
    orig = detector.DetectionDataset
    detector.DetectionDataset = lambda s, **k: (seen.append(len(s)), orig(s, **k))[1]
    try:
        detector.train_detector(pos, epochs=1, batch_size=2, pretrained=False, max_batches=1,
                                device="cpu", log=lambda *_: None, negatives=negs)
    finally:
        detector.DetectionDataset = orig
    assert seen == [5]
    alarms = detector.false_alarms_per_photo(model, negs)
    assert alarms >= 0
    text = detector.format_report(detector.build_report(detector.predict_samples(model, pos[:2])),
                                  false_alarms=alarms, test_scenes=2)
    assert "false alarms per background-only photo" in text and "2 different scene" in text


def _video(root, name, color, n=3):
    files = []
    for i in range(n):
        im = Image.new("RGB", (160, 120), color)
        im.putpixel((i, i), (255, 255, 255))
        files.append((f"vid_{name}__f{i}.jpg", color))
    return files


def test_same_background_videos_land_on_same_side(tmp_path):
    root = tmp_path / "det"
    room = (120, 100, 80)
    _folder(root, "own", "shoes", _video(root, "a", room), "owlv2", [("shoes", [0.1, 0.1, 0.5, 0.5])])
    _folder(root, "own", "keys", _video(root, "b", (122, 100, 78)), "owlv2", [("keys", [0.1, 0.1, 0.5, 0.5])])
    _folder(root, "own", "water_bottle", _video(root, "c", (10, 20, 200)), "owlv2",
            [("water_bottle", [0.1, 0.1, 0.5, 0.5])])
    _folder(root, "own", "together", _video(root, "d", (250, 240, 10)), "owlv2",
            [("keys", [0.1, 0.1, 0.5, 0.5])])
    samples, _ = dd.scan_detection(root)
    scenes = dd.scene_groups(samples)
    assert scenes["vid_a"] == scenes["vid_b"] != scenes["vid_c"]
    for seed in range(8):
        logs = []
        train, test = dd.split_detection(samples, 0.5, seed, scenes=scenes, log=logs.append)
        side = {g: "test" if any(s.group == g for s in test) else "train"
                for g in ("vid_a", "vid_b")}
        assert side["vid_a"] == side["vid_b"]
        assert len(train) + len(test) == len(samples)
    assert any("vid_a" in l and "vid_b" in l for l in logs)


def test_one_scene_falls_back_to_by_video(tmp_path):
    root = tmp_path / "det"
    _folder(root, "own", "shoes", _video(root, "a", (5, 5, 5)), "owlv2", [("shoes", [0.1, 0.1, 0.5, 0.5])])
    _folder(root, "own", "keys", _video(root, "b", (5, 5, 5)), "owlv2", [("keys", [0.1, 0.1, 0.5, 0.5])])
    samples, _ = dd.scan_detection(root)
    logs = []
    dd.split_detection(samples, 0.5, 0, scenes=dd.scene_groups(samples), log=logs.append)
    assert any("ONE scene" in l for l in logs)
