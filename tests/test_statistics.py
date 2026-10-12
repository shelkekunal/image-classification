import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from image_classifier.data.statistics import calculate_channel_statistics


def test_calculates_channel_statistics() -> None:
    images = torch.zeros((2, 3, 2, 2))
    images[:, 0] = 0.2
    images[:, 1] = 0.4
    images[:, 2] = 0.6

    dataloader = DataLoader(
        TensorDataset(images, torch.zeros(2, dtype=torch.long)),
        batch_size=1,
    )

    mean, std = calculate_channel_statistics(dataloader)

    torch.testing.assert_close(mean, torch.tensor([0.2, 0.4, 0.6]))
    torch.testing.assert_close(std, torch.zeros(3))


def test_rejects_invalid_image_shape() -> None:
    images = torch.zeros((2, 1, 2, 2))
    dataloader = DataLoader(
        TensorDataset(images, torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="shape"):
        calculate_channel_statistics(dataloader)


def test_rejects_empty_dataset() -> None:
    images = torch.empty((0, 3, 2, 2))
    dataloader = DataLoader(
        TensorDataset(images, torch.empty(0, dtype=torch.long)),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="empty dataset"):
        calculate_channel_statistics(dataloader)
