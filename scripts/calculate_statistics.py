import logging

from torch.utils.data import DataLoader
from torchvision import transforms

from scripts.download_dataset import download_cifar10
from image_classifier.data.statistics import calculate_channel_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    transform = transforms.ToTensor()
    train_dataset, _ = download_cifar10()

    train_dataset.transform = transform

    dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    mean, std = calculate_channel_statistics(dataloader)

    logger.info("Training channel mean: %s", mean.tolist())
    logger.info("Training channel std: %s", std.tolist())


if __name__ == "__main__":
    main()
