"""CIFAR-10 data loading.

`get_dataloaders` returns (train_loader, test_loader). Pass `fake=True` to use
synthetic images of the same shape, which needs no download (for tests and
quick smoke runs).
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
NUM_CLASSES = len(CIFAR10_CLASSES)
IMAGE_SHAPE = (3, 32, 32)

# Per-channel mean/std of the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def eval_transform():
    """Transform for test/inference images: to tensor + normalise."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )


def train_transform():
    """Transform for training images: light augmentation + normalise."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_datasets(root="data", fake=False, fake_size=64, download=True):
    """Return (train_dataset, test_dataset)."""
    if fake:
        train = datasets.FakeData(
            size=fake_size,
            image_size=IMAGE_SHAPE,
            num_classes=NUM_CLASSES,
            transform=train_transform(),
            random_offset=0,
        )
        test = datasets.FakeData(
            size=fake_size,
            image_size=IMAGE_SHAPE,
            num_classes=NUM_CLASSES,
            transform=eval_transform(),
            random_offset=1,
        )
        return train, test

    train = datasets.CIFAR10(
        root, train=True, download=download, transform=train_transform()
    )
    test = datasets.CIFAR10(
        root, train=False, download=download, transform=eval_transform()
    )
    return train, test


def get_dataloaders(
    root="data",
    batch_size=128,
    num_workers=2,
    fake=False,
    fake_size=64,
    download=True,
):
    """Return (train_loader, test_loader) for CIFAR-10 (or fake data)."""
    train, test = get_datasets(
        root=root, fake=fake, fake_size=fake_size, download=download
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
