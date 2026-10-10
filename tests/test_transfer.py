import torch
from PIL import Image

from claudeknows import evaluate, predict, train_objects, transfer


def _folder(root, per_group=3):
    """3 objects: web photos (own groups) + 3 phone-video 'clips' for two of them."""
    colors = {"water_bottle": (30, 60, 200), "keys": (200, 180, 20), "shoes": (180, 30, 30)}
    for cls, col in colors.items():
        d = root / "web" / cls
        d.mkdir(parents=True)
        for i in range(10):
            Image.new("RGB", (64, 48), col).save(d / f"id{i}.jpg")
    for cls in ("keys", "shoes"):
        d = root / "own" / cls
        d.mkdir(parents=True)
        for v in range(3):
            for f in range(per_group):
                Image.new("RGB", (64, 48), colors[cls]).save(d / f"vid_v{v}__f{f:04d}.jpg")
    return root


def test_model_has_one_output_per_object():
    m = transfer.build_model(3, pretrained=False)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 3)
    assert m.eval()(torch.zeros(1, 3, 224, 224)).shape == (1, 3)


def test_train_save_evaluate_predict_roundtrip(tmp_path, capsys):
    root = _folder(tmp_path / "objects")
    ckpt = tmp_path / "objects.pt"
    hist = train_objects.train_objects(
        root, epochs=2, head_epochs=1, batch_size=8, image_size=64, checkpoint=str(ckpt),
        pretrained=False, max_batches=2, device=torch.device("cpu"))
    assert len(hist) == 2 and ckpt.exists()
    state = torch.load(ckpt)
    assert state["class_names"] == ["water_bottle", "keys", "shoes"]

    res = train_objects.evaluate_objects(str(ckpt), root, device=torch.device("cpu"),
                                         log=lambda *_: None)
    assert res["total"] > 0 and ("own", "keys") in res["per_source"]
    assert ("own", "water_bottle") not in res["per_source"]  # no footage -> reported as missing

    model, names, size, _ = transfer.load_transfer(ckpt)
    r = transfer.predict_transfer(root / "web" / "keys" / "id0.jpg", model, names, size)
    assert r["label"] in names and 0 < r["confidence"] <= 1

    # the existing CLIs recognise the new checkpoint without extra flags
    predict.main([str(root / "web" / "keys" / "id0.jpg"), "--checkpoint", str(ckpt)])
    assert any(n in capsys.readouterr().out for n in names)
    evaluate.main(["--checkpoint", str(ckpt), "--objects-root", str(root), "--num-workers", "0"])
    assert "overall accuracy" in capsys.readouterr().out


def test_trains_with_only_internet_photos(tmp_path):
    root = _folder(tmp_path / "objects")
    import shutil
    shutil.rmtree(root / "own")
    hist = train_objects.train_objects(
        root, epochs=1, head_epochs=1, batch_size=8, image_size=64, checkpoint=None,
        pretrained=False, max_batches=1, device=torch.device("cpu"))
    assert len(hist) == 1
