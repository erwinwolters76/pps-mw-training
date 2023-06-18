from pathlib import Path

import xarray as xr  # type: ignore


def load_retrieval_database(
    db_file: Path,
    every_other: bool = False,
    dimension: str = "number_structures_db",
) -> xr.Dataset:
    """Load the retrieval database."""
    db = xr.load_dataset(db_file)
    if every_other:
        db = db.sel({dimension: db[dimension].values[0::2]})
    return db
