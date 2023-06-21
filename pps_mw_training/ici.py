from pathlib import Path

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
