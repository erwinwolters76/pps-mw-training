from typing import Any
from pathlib import Path
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


def get_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calculate std and mean for the input dataset to normalise"""

    stats = {}
    with xr.open_dataset(input_files[0]) as ds:
        for parameter in ds:
            data = ds[parameter].values
            stats[parameter] = {
                "n": np.sum(np.isfinite(data)),
                "sum": np.nansum(data),
                "sum_of_squares": np.nansum(data**2),
            }

    for file in input_files[1:]:
        with xr.open_dataset(file) as ds:
            for parameter in ds:
                data = ds[parameter].values
                stats[parameter]["n"] += np.sum(np.isfinite(data))
                stats[parameter]["sum"] += np.nansum(data)
                stats[parameter]["sum_of_squares"] += np.nansum(data**2)

    for p in input_params:
        parameter = p["name"]
        p["mean"] = stats[parameter]["sum"] / stats[parameter]["n"]
        p["std"] = np.sqrt(
            (stats[parameter]["sum_of_squares"] / stats[parameter]["n"])
            - (stats[parameter]["sum"] / stats[parameter]["n"]) ** 2
        )

    return input_params
