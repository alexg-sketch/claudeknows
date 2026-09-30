"""Training script: python -m claudeknows.train [--fake --max-batches 2]."""
import argparse

import torch
import torch.nn as nn

from .data import get_dataloaders
from .model import SmallCNN


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    """Train for one epoch; return (mean loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate_accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct, seen = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        seen += y.size(0)
    return correct / max(seen, 1)


def train(
    epochs=20,
    lr=1e-3,
    batch_size=128,
    root="data",
    fake=False,
    max_batches=None,
    checkpoint="checkpoint.pt",
    num_workers=2,
    device=None,
):
    """Train SmallCNN, save a checkpoint, and return the history list."""
    device = device or pick_device()
    train_loader, test_loader = get_dataloaders(
        root=root, batch_size=batch_size, fake=fake, num_workers=num_workers
    )
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches
        )
        scheduler.step()
        test_acc = evaluate_accuracy(model, test_loader, device, max_batches)
        history.append(
            {"epoch": epoch, "loss": loss, "train_acc": train_acc, "test_acc": test_acc}
        )
        print(
            f"epoch {epoch}/{epochs} loss={loss:.4f} "
            f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
        )

    if checkpoint:
        torch.save({"model_state": model.state_dict(), "history": history}, checkpoint)
        print(f"saved checkpoint to {checkpoint}")
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--root", default="data")
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--fake", action="store_true", help="use synthetic data")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args(argv)
    train(
        epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, root=a.root,
        fake=a.fake, max_batches=a.max_batches, checkpoint=a.checkpoint,
        num_workers=a.num_workers,
    )


if __name__ == "__main__":
    main()
