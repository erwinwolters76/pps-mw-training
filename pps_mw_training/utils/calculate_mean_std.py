from typing import Any
from pathlib import Path
import numpy as np  # type: ignore
from tqdm import tqdm
import xarray as xr  # type: ignore


def get_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calculate std and mean for the input dataset to normalise"""

    all_dict = {}
    with xr.open_dataset(input_files[0]) as ds:
        for key in ds.keys():
            all_dict[key] = {"n": 0, "x": 0, "x2": 0, "min": 1e10, "max": 0}

    for file in tqdm(input_files):
        with xr.open_dataset(file) as ds:
            ds = ds.sel(
                {
                    "npix": ds["nscan"].values[8:-8],
                    "nscan": ds["npix"].values[8:-8],
                }
            )
            for key in ds.keys():
                data = ds[key].values
                all_dict[key]["n"] += np.sum(np.isfinite(data))
                all_dict[key]["x"] += np.nansum(data)
                all_dict[key]["x2"] += np.nansum(data**2)

                if data.size > 0:
                    all_dict[key]["min"] = min(
                        all_dict[key]["min"], np.nanmin(data)
                    )
                    all_dict[key]["max"] = max(
                        all_dict[key]["max"], np.nanmax(data)
                    )
    for p in input_params:
        key = p["name"]
        p["mean"] = all_dict[key]["x"] / all_dict[key]["n"]
        p["std"] = np.sqrt(
            (all_dict[key]["x2"] / all_dict[key]["n"])
            - (all_dict[key]["x"] / all_dict[key]["n"]) ** 2
        )
        p["min"] = all_dict[key]["min"]
        p["max"] = all_dict[key]["max"]

    return input_params
