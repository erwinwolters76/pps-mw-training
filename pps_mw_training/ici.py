import os
from pathlib import Path

import xarray as xr  # type: ignore


RETRIEVAL_DB_FILE = Path(
    os.environ.get(
        "RETRIEVAL_DB_FILE",
        "/home/a002491/ici_retrieval_database.nc",
    )
)


def load_retrieval_database(
    db_file: Path = RETRIEVAL_DB_FILE,
    every_other: bool = False,
    dimension: str = "number_structures_db",
) -> xr.Dataset:
    """Load the retrieval database."""
    db = xr.load_dataset(db_file)
    if every_other:
        db = db.sel({dimension: db[dimension].values[0::2]})
    return db
