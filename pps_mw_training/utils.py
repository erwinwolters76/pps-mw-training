from tensorflow.data import Dataset
from typing import Tuple

import numpy as np  # type: ignore


def to_numpy_arrays(
    dataset: Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract observation and state arrays from dataset."""
    observation, state = tuple(zip(*dataset))
    return np.array(observation), np.array(state)


def split_dataset(
    dataset: Dataset,
    train_fraction: int,
    validation_fraction: int,
    test_fraction: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into a training, validation, and test dataset."""
    n_samples = len(dataset)
    train_size = int(train_fraction * n_samples)
    val_size = int(validation_fraction * n_samples)
    test_size = int(test_fraction * n_samples)
    training_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    validation_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return training_dataset, test_dataset, validation_dataset


def create_simple_training_dataset(
    n_samples: int,
    n_channels: int,
    n_params: int,
) -> Dataset:
    """Create a simple 'toy' training dataset."""
    state = np.stack(
        [i + 0.5 * np.random.randn(n_samples) for i in range(n_params)]
    ).T
    observation = np.zeros((n_samples, n_channels))
    for i in range(n_channels):
        noise = 0.01 * np.random.randn(n_samples)
        weights = np.random.rand(n_params)
        weighted = np.matmul(state, weights / np.sum(weights))
        observation[:, i] = weighted + np.sin(weighted) + noise
    return Dataset.from_tensor_slices((observation, state))
