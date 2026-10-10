from scripts.download_dataset import download_cifar10


def test_cifar10_dataset_sizes() -> None:
    train_dataset, test_dataset = download_cifar10()

    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000


def test_cifar10_image_and_label_format() -> None:
    train_dataset, _ = download_cifar10()

    image, label = train_dataset[0]

    assert image.size == (32, 32)
    assert image.mode == "RGB"
    assert isinstance(label, int)
    assert 0 <= label < 10
