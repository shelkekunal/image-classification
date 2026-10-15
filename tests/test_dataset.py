import torch

from image_classifier.data.dataset import load_cifar10_datasets


def test_load_cifar10_datasets() -> None:
    train_dataset, test_dataset = load_cifar10_datasets()

    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000

    image, label = train_dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 10


def test_test_dataset_uses_preprocessing() -> None:
    _, test_dataset = load_cifar10_datasets()

    image, label = test_dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    assert 0 <= label < 10
