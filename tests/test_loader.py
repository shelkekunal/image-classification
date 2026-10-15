import torch
from torch.utils.data import TensorDataset

from image_classifier.data.loader import create_dataloader


def test_dataloader_creates_batches() -> None:
    images = torch.randn(100, 3, 32, 32)
    labels = torch.zeros(100, dtype=torch.long)
    dataset = TensorDataset(images, labels)

    loader = create_dataloader(dataset, batch_size=64)

    batches = list(loader)

    assert len(batches) == 2
    assert batches[0][0].shape == (64, 3, 32, 32)
    assert batches[1][0].shape == (36, 3, 32, 32)


def test_dataloader_can_shuffle() -> None:
    images = torch.arange(100).float().reshape(100, 1)
    labels = torch.zeros(100, dtype=torch.long)
    dataset = TensorDataset(images, labels)

    loader = create_dataloader(dataset, batch_size=100, shuffle=True)
    shuffled_images, _ = next(iter(loader))

    assert sorted(shuffled_images.flatten().tolist()) == list(
        range(100)
    )
