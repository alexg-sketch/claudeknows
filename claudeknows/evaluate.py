"""Evaluation: python -m claudeknows.evaluate --checkpoint checkpoint.pt [--fake]."""
import argparse

import torch

from .data import CIFAR10_CLASSES, NUM_CLASSES, get_dataloaders
from .model import SmallCNN
from .train import pick_device


def load_model(checkpoint, device=None):
    """Load a SmallCNN from a checkpoint saved by `claudeknows.train`."""
    device = device or pick_device()
    state = torch.load(checkpoint, map_location=device)
    model = SmallCNN().to(device)
    model.load_state_dict(state["model_state"])
    return model.eval()


@torch.no_grad()
def confusion_matrix(model, loader, device, max_batches=None):
    """Return a (classes x classes) tensor; rows = true label, cols = predicted."""
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
    return cm


def summarize(cm):
    """Return dict with overall accuracy and per-class accuracy (by name)."""
    total = cm.sum().item()
    overall = cm.diag().sum().item() / max(total, 1)
    per_class = {}
    for i, name in enumerate(CIFAR10_CLASSES):
        n = cm[i].sum().item()
        per_class[name] = cm[i, i].item() / n if n else float("nan")
    return {"accuracy": overall, "per_class": per_class, "total": total}


def format_report(cm):
    s = summarize(cm)
    lines = [f"overall accuracy: {s['accuracy']:.3f} ({s['total']} images)", "per-class accuracy:"]
    lines += [f"  {n:<11}{a:.3f}" for n, a in s["per_class"].items()]
    lines.append("confusion matrix (rows=true, cols=predicted):")
    lines.append(" " * 12 + " ".join(f"{n[:5]:>5}" for n in CIFAR10_CLASSES))
    for name, row in zip(CIFAR10_CLASSES, cm.tolist()):
        lines.append(f"{name:<11} " + " ".join(f"{v:>5}" for v in row))
    return "\n".join(lines)


def evaluate(checkpoint, root="data", fake=False, batch_size=256, max_batches=None,
             num_workers=2, device=None):
    device = device or pick_device()
    _, test_loader = get_dataloaders(
        root=root, batch_size=batch_size, fake=fake, num_workers=num_workers
    )
    model = load_model(checkpoint, device)
    cm = confusion_matrix(model, test_loader, device, max_batches)
    print(format_report(cm))
    return cm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args(argv)
    evaluate(a.checkpoint, a.root, a.fake, a.batch_size, a.max_batches, a.num_workers)


if __name__ == "__main__":
    main()
