import pytest
import torch
from PIL import Image

from image_classifier.data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_base_transform,
)


def test_base_transform_returns_normalized_tensor() -> None:
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))

    transformed_image = get_base_transform()(image)

    assert transformed_image.shape == (3, 32, 32)
    assert transformed_image.dtype == torch.float32
    assert torch.isfinite(transformed_image).all()

    expected = (
        (128 / 255 - torch.tensor(CIFAR10_MEAN))
        / torch.tensor(CIFAR10_STD)
    )

    torch.testing.assert_close(
        transformed_image[:, 0, 0],
        expected,
    )


def test_base_transform_rejects_invalid_image_input() -> None:
    transform = get_base_transform()

    with pytest.raises((TypeError, ValueError)):
        transform("not an image")
