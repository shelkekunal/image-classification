from pathlib import Path
from unittest.mock import patch

from scripts.download_dataset import download_cifar10


def test_download_cifar10_creates_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    expected_train = object()
    expected_test = object()

    with patch("scripts.download_dataset.CIFAR10") as mock_cifar10:
        mock_cifar10.side_effect = [expected_train, expected_test]

        train_dataset, test_dataset = download_cifar10(data_dir)

    assert data_dir.is_dir()
    assert mock_cifar10.call_count == 2

    assert mock_cifar10.call_args_list[0].kwargs == {
        "root": str(data_dir),
        "train": True,
        "download": True,
    }

    assert mock_cifar10.call_args_list[1].kwargs == {
        "root": str(data_dir),
        "train": False,
        "download": True,
    }

    assert train_dataset is expected_train
    assert test_dataset is expected_test