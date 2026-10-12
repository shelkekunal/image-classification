import matplotlib.pyplot as plt

from scripts.download_dataset import download_cifar10

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

    figure, axes = plt.subplots(2, 5, figsize=(12, 6))

    for class_index, axis in enumerate(axes.flat):
        image_index = next(
            index
            for index, label in enumerate(train_dataset.targets)
            if label == class_index
        )

        image, label = train_dataset[image_index]

        axis.imshow(image, interpolation="nearest")
        axis.set_title(CIFAR10_CLASSES[label])
        axis.axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
