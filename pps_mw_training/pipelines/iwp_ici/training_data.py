from pathlib import Path
from typing import cast, Dict, List, Tuple, Union

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


DB_SURFACE_TYPES = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    15: 4,
}


def load_retrieval_database(
    db_file: Path,
    every_other: bool = False,
    dimension: str = "number_structures_db",
) -> xr.Dataset:
    """Load the retrieval database."""
    db = xr.load_dataset(db_file)
    if every_other:
        db = db.sel({dimension: db[dimension].values[0::2]})
    db["SurfType"].values = adjust_surface_type(
        db["SurfType"].values
    )
    return db


def adjust_surface_type(surface_type: np.ndarray) -> np.ndarray:
    """Adjust surface type."""
    adjusted = np.zeros_like(surface_type)
    for old_value, new_value in DB_SURFACE_TYPES.items():
        adjusted[surface_type == old_value] = new_value
    return adjusted


def add_noise(
    dataset: xr.Dataset,
    params: List[str],
    sigma: float,
) -> xr.Dataset:
    """Add normal distributed noise to given params."""
    for param in params:
        dataset[param].values += sigma * np.random.randn(dataset[param].size)
    return dataset


def split_dataset(
    dataset: xr.Dataset,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    dimension: str = "number_structures_db",
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Split dataset into three parts."""
    n_samples = dataset[dimension].size
    fractions = [train_fraction, validation_fraction, test_fraction]
    limits = np.cumsum([int(f * n_samples) for f in fractions])
    return (
        dataset.sel(
            {dimension: dataset[dimension].values[0: limits[0]]}
        ),
        dataset.sel(
            {dimension: dataset[dimension].values[limits[0]: limits[1]]}
        ),
        dataset.sel(
            {dimension: dataset[dimension].values[limits[1]: limits[2]]}
        ),
    )


def get_training_data(
    ici_db_file: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    input_params: List[Dict[str, Union[str, float]]],
    noise: float,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Get training data."""
    full_dataset = load_retrieval_database(ici_db_file)
    params = [cast(str, p["name"]) for p in input_params]
    full_dataset = add_noise(
        full_dataset,
        params=[p for p in params if p.startswith("DTB")],
        sigma=noise,
    )
    return split_dataset(
        full_dataset,
        train_fraction,
        validation_fraction,
        test_fraction,
    )
