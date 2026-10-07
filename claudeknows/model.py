"""Small CNN for CIFAR-10 classification."""
import torch.nn as nn

from .data import NUM_CLASSES


def _block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Three conv blocks (32x32 -> 4x4) followed by a linear classifier."""

    def __init__(self, num_classes=NUM_CLASSES, width=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width),
            _block(width, width * 2),
            _block(width * 2, width * 4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4 * 4 * 4, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
