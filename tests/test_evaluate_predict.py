import torch
from PIL import Image

from claudeknows.data import NUM_CLASSES, get_dataloaders
from claudeknows.evaluate import confusion_matrix, format_report, load_model, summarize
from claudeknows.predict import predict_image
from claudeknows.train import train


def _checkpoint(tmp_path):
    path = str(tmp_path / "ck.pt")
    train(epochs=1, batch_size=16, fake=True, max_batches=2, checkpoint=path,
          num_workers=0, device=torch.device("cpu"))
    return path


def test_summarize_known_matrix():
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm[0, 0], cm[0, 1], cm[1, 1] = 3, 1, 4
    s = summarize(cm)
    assert abs(s["accuracy"] - 7 / 8) < 1e-9
    assert s["per_class"]["airplane"] == 0.75
    assert s["per_class"]["automobile"] == 1.0
    assert "overall accuracy" in format_report(cm)


def test_confusion_matrix_counts(tmp_path):
    model = load_model(_checkpoint(tmp_path), torch.device("cpu"))
    _, loader = get_dataloaders(batch_size=16, fake=True, num_workers=0)
    cm = confusion_matrix(model, loader, torch.device("cpu"), max_batches=2)
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert cm.sum().item() == 32


def test_predict_image(tmp_path):
    model = load_model(_checkpoint(tmp_path), torch.device("cpu"))
    img = tmp_path / "x.png"
    Image.new("RGB", (50, 40), (200, 30, 30)).save(img)
    r = predict_image(str(img), model)
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["label"] == r["top"][0][0]
    assert len(r["top"]) == 3
