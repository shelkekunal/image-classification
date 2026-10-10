from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_base_transform() -> transforms.Compose:
    """Convert CIFAR-10 images to normalized tensors."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )
