from pathlib import Path
import logging

from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the dataset will be stored
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)

def download_cifar10(data_dir : Path = DATA_DIR) -> tuple[CIFAR10, CIFAR10]:
    """Download and return the CIFAR10 training and testing datasets"""
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Training dataset...")
    train_dataset = CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
    )

    logger.info("Downloading testing datasets...")
    test_dataset = CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
    )

    logger.info("CIFAR10 dataset are ready")

    return train_dataset, test_dataset

if __name__ == "__main__":
    download_cifar10()