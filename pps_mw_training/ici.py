import os
from pathlib import Path
from tensorflow.data import Dataset

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


RETRIEVAL_DB_FILE = Path(
    os.environ.get(
        "RETRIEVAL_DB_FILE",
        "/home/a002491/ici_retrieval_database.nc",
    )
)
CHANNEL_PARAMS = [
    'DTB_ICI_DB_ICI_01V',
    'DTB_ICI_DB_ICI_02V',
    'DTB_ICI_DB_ICI_03V',
    'DTB_ICI_DB_ICI_04V',
    'DTB_ICI_DB_ICI_04H',
    'DTB_ICI_DB_ICI_05V',
    'DTB_ICI_DB_ICI_06V',
    'DTB_ICI_DB_ICI_07V',
    'DTB_ICI_DB_ICI_08V',
    'DTB_ICI_DB_ICI_09V',
    'DTB_ICI_DB_ICI_10V',
    'DTB_ICI_DB_ICI_11V',
    'DTB_ICI_DB_ICI_11H',
    'SurfType',
    'SurfPres',
    'SurfTemp',
    'SurfWind',
]
STATE_PARAMS = [
    'TCWV',
    'LWP',
    'RWP',
    'IWP',
    'Zmean',
    'Dmean',
]
MIN_VALUE = 1e-6


def load_retrieval_database(
    db_file: Path = RETRIEVAL_DB_FILE,
    every_other: bool = True,
) ->Dataset:
    """Load the retrieval database."""
    db = xr.load_dataset(db_file)
    if every_other:
        # remove every second state
        db = db.sel(
            number_structures_db=db.number_structures_db.values[0::5]
        )
    observation = np.vstack(
        list([scale(param, db[param].values)] for param in CHANNEL_PARAMS)
    ).T
    state = np.vstack(
        list([scale(param, db[param].values)] for param in STATE_PARAMS)
    ).T
    return Dataset.from_tensor_slices((observation, state))


def scale(param: str, data: np.ndarray) -> np.ndarray:
    if param in ["LWP", "RWP", "IWP"]:
        data[data < MIN_VALUE] = MIN_VALUE
        data = np.log(data)
    delta = np.max(data) - np.min(data)
    return  2 * (data - np.min(data)) / delta - 1
