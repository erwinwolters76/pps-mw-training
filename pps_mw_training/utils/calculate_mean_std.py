from typing import Any
from pathlib import Path

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


def get_stats(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Calculate standard deviation and mean for the input dataset to normalize.
    """
    return get_std_mean(input_files, input_params)


def get_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Calculate intermediate statistics to calculate mean and std
    and then update the input_params dict
    """

    stats = {
        p["name"]: {"n": 0, "sum": 0.0, "sum_of_squares": 0.0}
        for p in input_params
    }
    for file in input_files:
        with xr.open_dataset(file) as ds:
            for parameter in ds:
                if parameter not in stats:
                    continue
                data = ds[parameter].values
                stats[parameter]["n"] += np.sum(np.isfinite(data))
                stats[parameter]["sum"] += np.nansum(data)
                stats[parameter]["sum_of_squares"] += np.nansum(data**2)
    return [
        param | calculate_std_mean(stats[param["name"]])
        for param in input_params
        if param["name"] in stats
    ]


def calculate_std_mean(stats: dict[str, float]) -> dict[str, float]:
    """
    Calculate mean and standard deviation from intermediate stats
    """
    mean = stats["sum"] / stats["n"]
    std = np.sqrt((stats["sum_of_squares"] / stats["n"]) - (mean**2))
    return {"mean": mean, "std": std}
