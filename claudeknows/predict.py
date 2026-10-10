"""Predict the label of one image: python -m claudeknows.predict IMAGE [--checkpoint checkpoint.pt]."""
import argparse

import torch
from PIL import Image

from .data import CIFAR10_CLASSES, eval_transform
from . import transfer
from .evaluate import load_model
from .train import pick_device


@torch.no_grad()
def predict_image(image, model, device=None, top_k=3):
    """Classify a PIL image (or path). Returns {'label', 'confidence', 'top'}."""
    device = device or next(model.parameters()).device
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB").resize((32, 32))
    x = eval_transform()(image).unsqueeze(0).to(device)
    probs = torch.softmax(model.eval()(x), dim=1)[0].cpu()
    conf, idx = probs.topk(min(top_k, len(CIFAR10_CLASSES)))
    top = [(CIFAR10_CLASSES[i], c) for i, c in zip(idx.tolist(), conf.tolist())]
    return {"label": top[0][0], "confidence": top[0][1], "top": top}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--checkpoint", default="checkpoint.pt")
    a = p.parse_args(argv)
    device = pick_device()
    state = torch.load(a.checkpoint, map_location=device)
    if transfer.is_transfer_checkpoint(state):  # new-objects model (class names inside)
        model, names, size, _ = transfer.load_transfer(a.checkpoint, device)
        r = transfer.predict_transfer(a.image, model, names, size)
    else:
        model = load_model(a.checkpoint, device)
        r = predict_image(a.image, model)
    print(f"{r['label']} ({r['confidence']:.1%})")
    for name, c in r["top"]:
        print(f"  {name:<11}{c:.1%}")


if __name__ == "__main__":
    main()
