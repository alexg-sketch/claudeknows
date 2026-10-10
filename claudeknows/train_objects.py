"""Train/evaluate the new-objects model.

    python -m claudeknows.train_objects --root data/objects --epochs 8
    python -m claudeknows.train_objects --evaluate --checkpoint objects.pt

Uses the internet photos (`web/`) and Alex's own footage (`own/`) together; a
missing source is fine. Held-out test photos are chosen by video/photo (never by
frame) and results are reported per object AND per source.
"""
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import transfer
from .objects import ObjectDataset, describe, scan, split_by_group
from .train import pick_device


def make_loaders(root, image_size, batch_size, seed=0, num_workers=0, test_frac=0.2):
    classes, samples = scan(root)
    present = {s.label for s in samples}
    if len(present) < 2:
        raise SystemExit(f"Need photos for at least 2 objects under {root}/web or {root}/own "
                         f"(found {len(present)}). Run claudeknows.openimages first.")
    train, test = split_by_group(samples, test_frac, seed)
    train_dl = DataLoader(ObjectDataset(train, transfer.train_transform(image_size)),
                          batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(ObjectDataset(test, transfer.eval_transform(image_size)),
                         batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return classes, train, test, train_dl, test_dl


def _run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    loss_sum = correct = seen = 0
    with torch.set_grad_enabled(optimizer is not None):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)
    return loss_sum / max(seen, 1), correct / max(seen, 1)


def train_objects(root="data/objects", epochs=8, head_epochs=1, lr=1e-3, batch_size=32,
                  image_size=transfer.IMAGE_SIZE, checkpoint="objects.pt", pretrained=True,
                  max_batches=None, seed=0, num_workers=0, device=None):
    """Fine-tune the network and save a checkpoint. Returns the history list.

    The first `head_epochs` epochs train only the new final layer, then the
    whole network is fine-tuned at a lower learning rate.
    """
    torch.manual_seed(seed)
    device = device or pick_device()
    classes, train, test, train_dl, test_dl = make_loaders(
        root, image_size, batch_size, seed, num_workers)
    print(describe(classes, train, test))
    model = transfer.build_model(len(classes), pretrained).to(device)
    counts = Counter(s.label for s in train)
    weights = torch.tensor([1.0 / max(counts.get(i, 0), 1) for i in range(len(classes))])
    criterion = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device))

    history, optimizer = [], None
    for epoch in range(1, epochs + 1):
        if epoch == 1 or epoch == head_epochs + 1:
            frozen = epoch <= head_epochs
            transfer.set_backbone_trainable(model, not frozen)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr if frozen else lr / 5, weight_decay=1e-4)
        loss, tr_acc = _run_epoch(model, train_dl, criterion, device, optimizer, max_batches)
        _, te_acc = _run_epoch(model, test_dl, criterion, device, None, max_batches)
        history.append({"epoch": epoch, "loss": loss, "train_acc": tr_acc, "test_acc": te_acc})
        print(f"epoch {epoch}/{epochs} loss={loss:.4f} train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")
    if checkpoint:
        transfer.save_checkpoint(checkpoint, model, classes, history, image_size, pretrained, seed)
        print(f"saved checkpoint to {checkpoint}")
    return history


@torch.no_grad()
def evaluate_objects(checkpoint, root="data/objects", batch_size=32, show_wrong=15,
                     num_workers=0, device=None, log=print):
    """Report accuracy overall, per object, per source, plus the photos it got wrong."""
    device = device or pick_device()
    model, classes, image_size, state = transfer.load_transfer(checkpoint, device)
    scanned_classes, samples = scan(root, classes)
    _, test = split_by_group(samples, seed=state.get("split_seed", 0))
    if not test:
        log("No held-out test photos found.")
        return None
    loader = DataLoader(ObjectDataset(test, transfer.eval_transform(image_size)),
                        batch_size=batch_size, num_workers=num_workers)
    preds = []
    for x, _ in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
    n = len(classes)
    cm = torch.zeros(n, n, dtype=torch.long)
    for s, p in zip(test, preds):
        cm[s.label, p] += 1
    ok = [s.label == p for s, p in zip(test, preds)]
    result = {"accuracy": sum(ok) / len(ok), "total": len(ok), "per_class": {}, "per_source": {}}
    log(f"overall accuracy: {result['accuracy']:.3f} ({len(ok)} held-out photos)")
    log("per object:")
    for i, name in enumerate(classes):
        row = cm[i].sum().item()
        result["per_class"][name] = cm[i, i].item() / row if row else float("nan")
        log(f"  {name:<14}{result['per_class'][name]:.3f}  ({row} photos)")
    log("per source (web = internet photos, own = Alex's footage):")
    for source in ("web", "own"):
        for i, name in enumerate(classes):
            idx = [k for k, s in enumerate(test) if s.source == source and s.label == i]
            if idx:
                acc = sum(ok[k] for k in idx) / len(idx)
                result["per_source"][(source, name)] = acc
                log(f"  {source:<4}{name:<14}{acc:.3f}  ({len(idx)} photos)")
            else:
                log(f"  {source:<4}{name:<14}no test photos")
    log("confusion matrix (rows=true, cols=predicted): " + ", ".join(classes))
    for name, row in zip(classes, cm.tolist()):
        log(f"  {name:<14}" + " ".join(f"{v:>5}" for v in row))
    wrong = [(s, classes[p]) for s, p, k in zip(test, preds, ok) if not k]
    result["wrong"] = wrong
    if wrong:
        log(f"photos it got wrong ({len(wrong)}; showing {min(show_wrong, len(wrong))}):")
        for s, guess in wrong[:show_wrong]:
            log(f"  {s.path}  true={classes[s.label]}  guessed={guess}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/objects")
    p.add_argument("--checkpoint", default="objects.pt")
    p.add_argument("--evaluate", action="store_true", help="evaluate instead of training")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--head-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=transfer.IMAGE_SIZE)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="random start (offline tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args(argv)
    if a.evaluate:
        evaluate_objects(a.checkpoint, a.root, a.batch_size, num_workers=a.num_workers)
    else:
        train_objects(a.root, a.epochs, a.head_epochs, a.lr, a.batch_size, a.image_size,
                      a.checkpoint, not a.no_pretrained, a.max_batches, a.seed, a.num_workers)


if __name__ == "__main__":
    main()
