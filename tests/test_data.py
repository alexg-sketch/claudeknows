import torch

from claudeknows.data import (
    CIFAR10_CLASSES,
    IMAGE_SHAPE,
    NUM_CLASSES,
    get_dataloaders,
)


def test_class_names():
    assert NUM_CLASSES == 10
    assert len(set(CIFAR10_CLASSES)) == 10
    assert CIFAR10_CLASSES[0] == "airplane"
    assert CIFAR10_CLASSES[-1] == "truck"


def test_fake_loaders_batch_shapes():
    train_loader, test_loader = get_dataloaders(
        batch_size=8, num_workers=0, fake=True, fake_size=16
    )
    for loader in (train_loader, test_loader):
        images, labels = next(iter(loader))
        assert images.shape == (8, *IMAGE_SHAPE)
        assert images.dtype == torch.float32
        assert labels.shape == (8,)
        assert labels.min() >= 0 and labels.max() < NUM_CLASSES


def test_fake_loader_sizes():
    train_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, fake=True, fake_size=10
    )
    assert len(train_loader.dataset) == 10
    assert len(test_loader) == 3  # 10 images / batch of 4, rounded up


def test_images_are_normalised():
    _, test_loader = get_dataloaders(
        batch_size=16, num_workers=0, fake=True, fake_size=16
    )
    images, _ = next(iter(test_loader))
    # After normalisation, pixel values are no longer confined to [0, 1].
    assert images.min() < 0


def test_test_split_is_deterministic():
    _, loader_a = get_dataloaders(batch_size=4, num_workers=0, fake=True)
    _, loader_b = get_dataloaders(batch_size=4, num_workers=0, fake=True)
    a, _ = next(iter(loader_a))
    b, _ = next(iter(loader_b))
    assert torch.equal(a, b)
