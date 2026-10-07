"""Build the photo-upload web page: python -m claudeknows.export_web [--checkpoint ...].

Folds batch-norm into the conv weights, packs all weights into the page as
base64, and writes a single self-contained `web/index.html` that runs the
model in the browser (no server, no libraries).
"""
import argparse
import base64
import json
from pathlib import Path

import torch

from .data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from .model import SmallCNN

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DEFAULT_CHECKPOINT = ROOT / "models" / "cifar10_smallcnn.pt"


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallCNN()
    model.load_state_dict(state["model_state"])
    return model.eval(), state.get("history", [])


def _fold(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w, b


@torch.no_grad()
def pack_weights(model):
    """Return (meta, float32 array) in the order web/classifier.js reads them."""
    parts, channels = [], [3]
    for block in model.features:
        for conv, bn in ((block[0], block[1]), (block[3], block[4])):
            w, b = _fold(conv, bn)
            parts += [w.flatten(), b.flatten()]
            channels.append(conv.out_channels)
    fc = model.classifier[2]
    parts += [fc.weight.flatten(), fc.bias.flatten()]
    flat = torch.cat(parts).numpy().astype("<f4")
    meta = {
        "channels": channels,
        "fcIn": fc.in_features,
        "numClasses": fc.out_features,
        "classes": list(CIFAR10_CLASSES),
        "mean": list(CIFAR10_MEAN),
        "std": list(CIFAR10_STD),
    }
    return meta, flat


def build_page(checkpoint=DEFAULT_CHECKPOINT):
    """Return (full_html, fragment). The fragment has no <html>/<head>/<body>."""
    model, history = load_checkpoint(checkpoint)
    meta, flat = pack_weights(model)
    if history:
        meta["testAccuracy"] = history[-1]["test_acc"]
        meta["epochs"] = history[-1]["epoch"]
    fragment = (
        (WEB / "template.html")
        .read_text()
        .replace("/*__CLASSIFIER_JS__*/", (WEB / "classifier.js").read_text())
        .replace("__MODEL_META__", json.dumps(meta))
        .replace("__MODEL_WEIGHTS__", base64.b64encode(flat.tobytes()).decode())
    )
    full = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n"
    )
    return full, fragment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--out", default=str(WEB / "index.html"))
    p.add_argument("--fragment-out", help="also write the page without the html/head/body wrapper")
    a = p.parse_args(argv)
    full, fragment = build_page(a.checkpoint)
    Path(a.out).write_text(full)
    print(f"wrote {a.out} ({len(full) / 1e6:.1f} MB)")
    if a.fragment_out:
        Path(a.fragment_out).write_text(fragment)
        print(f"wrote {a.fragment_out}")


if __name__ == "__main__":
    main()
