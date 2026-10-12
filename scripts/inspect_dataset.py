import logging

from scripts.download_dataset import download_cifar10
from image_classifier.data.inspection import get_class_distribution

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
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
]


def main() -> None:
    train_dataset, _ = download_cifar10()

    distribution = get_class_distribution(
        train_dataset.targets,
        num_classes=len(CIFAR10_CLASSES),
    )

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Class distribution:")

    for class_index, count in distribution.items():
        logger.info("%s: %d", CIFAR10_CLASSES[class_index], count)


if __name__ == "__main__":
    main()
