import pytest

from image_classifier.data.inspection import get_class_distribution


def test_get_class_distribution_counts_labels() -> None:
    targets = [0, 1, 0, 2, 1, 0]

    result = get_class_distribution(targets, num_classes=3)

    assert result == {0: 3, 1: 2, 2: 1}


def test_get_class_distribution_includes_empty_classes() -> None:
    result = get_class_distribution([1, 1], num_classes=3)

    assert result == {0: 0, 1: 2, 2: 0}


def test_get_class_distribution_rejects_invalid_labels() -> None:
    with pytest.raises(ValueError, match="valid class indices"):
        get_class_distribution([0, 10])


def test_get_class_distribution_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        get_class_distribution([0], num_classes=0)
