"""Transfer learning for the new objects (water bottle, keys, shoes).

    python -m claudeknows.train_objects --root data/objects --epochs 8

A small pretrained network (MobileNetV3-small) gets a new final layer with one
output per object folder. The class names are saved inside the checkpoint, so
`predict` and `evaluate` need no extra settings. Pretrained weights download in
Colab; in offline sandboxes use `--no-pretrained` (tests do).
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ARCH = "mobilenet_v3_small"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(num_classes, pretrained=True):
    """MobileNetV3-small with its last layer replaced by `num_classes` outputs."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable):
    """Freeze/unfreeze everything except the classifier head."""
    for p in model.features.parameters():
        p.requires_grad = trainable


def eval_transform(image_size=IMAGE_SIZE):
    # Squash (not centre-crop) so a long object at the edge is never cut off.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(image_size=IMAGE_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def is_transfer_checkpoint(state):
    return isinstance(state, dict) and "class_names" in state


def save_checkpoint(path, model, class_names, history, image_size, pretrained, seed):
    torch.save({
        "arch": ARCH, "class_names": list(class_names), "image_size": image_size,
        "model_state": model.state_dict(), "history": history,
        "pretrained_start": bool(pretrained), "split_seed": seed,
    }, path)


def load_transfer(checkpoint, device="cpu"):
    """Return (model, class_names, image_size) from a transfer checkpoint."""
    state = torch.load(checkpoint, map_location=device)
    if not is_transfer_checkpoint(state):
        raise ValueError(f"{checkpoint} is not a transfer-learning checkpoint")
    model = build_model(len(state["class_names"]), pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    return model.eval(), tuple(state["class_names"]), state["image_size"], state


@torch.no_grad()
def predict_transfer(image, model, class_names, image_size=IMAGE_SIZE, top_k=3, device=None):
    device = device or next(model.parameters()).device
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    x = eval_transform(image_size)(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model.eval()(x), dim=1)[0].cpu()
    conf, idx = probs.topk(min(top_k, len(class_names)))
    top = [(class_names[i], c) for i, c in zip(idx.tolist(), conf.tolist())]
    return {"label": top[0][0], "confidence": top[0][1], "top": top}
