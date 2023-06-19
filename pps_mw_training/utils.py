from typing import List, Tuple

from xarray import Dataset  # type: ignore
import numpy as np  # type: ignore


def as_array(
    data: Dataset,
    params: List[str],
) -> np.ndarray:
    """Get dataset as an array."""
    return np.column_stack([data[param].values for param in params])


def add_noise(
    dataset: Dataset,
    params: List[str],
    sigma: float,
) -> Dataset:
    """Add normal distributed noise to given params."""
    for param in params:
        dataset[param].values += sigma * np.random.randn(dataset[param].size)
    return dataset


def split_dataset(
    dataset: Dataset,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    dimension: str = "number_structures_db",
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into a three parts."""
    n_samples = dataset[dimension].size
    train_limit = int(train_fraction * n_samples)
    test_limit = train_limit + int(test_fraction * n_samples)
    val_limit = test_limit + int(validation_fraction * n_samples)
    return (
        dataset.sel(
            {dimension: dataset[dimension].values[0: train_limit]}
        ),
        dataset.sel(
            {dimension: dataset[dimension].values[train_limit: test_limit]}
        ),
        dataset.sel(
            {dimension: dataset[dimension].values[test_limit: val_limit]}
        ),
    )
