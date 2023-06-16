from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from xarray import Dataset  # type: ignore
import numpy as np  # type: ignore


MIN_VALUE = 1e-6


@dataclass
class Scaler:
    """Scaler object."""
    params: List[Dict[str, Any]]

    def as_numpy_array(
        self,
        data: Dataset,
    ) -> np.ndarray:
        """Return data as numpy array."""
        return np.vstack(
            [data[param["name"]].values for param in self.params]
        ).T

    def apply(
        self,
        data: Dataset,
        as_numpy_array: bool = True,
    ) -> Union[np.ndarray, Dataset]:
        """Apply forward scaling."""
        for param in self.params:
            data[param["name"]].values = self._apply(
                data[param["name"]].values,
                param["min"],
                param["max"],
                param["scale"],
            )
        if as_numpy_array:
            return self.as_numpy_array(data)
        return data

    def reverse(
        self,
        data: Dataset,
        as_numpy_array: bool = False,
    ) -> Union[np.ndarray, Dataset]:
        """Apply reversed scaling."""
        for param in self.params:
            data[param["name"]].values = self._reverse(
                data[param["name"]].values,
                param["min"],
                param["max"],
                param["scale"],
            )
        if as_numpy_array:
            return self.as_numpy_array(data)
        return data

    @staticmethod
    def _apply(
        data: np.ndarray,
        min_value: float,
        max_value: float,
        scale: str,
    ) -> np.ndarray:
        """Apply forward scaling."""
        if scale == "log":
            min_value = max(min_value, MIN_VALUE)
            data[data == 0] = min_value
            min_value = np.log(min_value)
            max_value = np.log(max_value)
            data = np.log(data)
        return 2 * (data - min_value) / (max_value - min_value) - 1

    @staticmethod
    def _reverse(
        data: np.ndarray,
        min_value: float,
        max_value: float,
        scale: str,
    ) -> np.ndarray:
        """Apply reversed scaling."""
        if scale == "log":
            min_value = max(min_value, MIN_VALUE)
            min_value = np.log(min_value)
            max_value = np.log(max_value)
        data = 0.5 * (data + 1.) * (max_value - min_value) + min_value
        if scale == "log":
            data = np.exp(data)
        return data


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
