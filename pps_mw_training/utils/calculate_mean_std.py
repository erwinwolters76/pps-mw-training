from typing import Any
from pathlib import Path

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


def get_std_mean(
    input_files: list[Path], params: list[str]
) -> dict[str, dict[str, float]]:
    """Get standard deviation and mean for given parameters of dataset."""

    def _get_std_mean(n, sum, sum_of_squares):
        return {
            "mean": sum / n,
            "std": np.sqrt((sum_of_squares / n) - ((sum / n) ** 2)),
        }

    stats = {p: {"n": 0, "sum": 0.0, "sum_of_squares": 0.0} for p in params}
    for file in input_files:
        with xr.open_dataset(file) as ds:
            for p in params:
                stats[p]["n"] += np.count_nonzero(ds[p].values)
                stats[p]["sum"] += np.nansum(ds[p].values)
                stats[p]["sum_of_squares"] += np.nansum(ds[p].values ** 2)

    return {p: _get_std_mean(**stat) for p, stat in stats.items()}


def update_params(
    params: list[dict[str, Any]],
    data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Update parameters with desired data."""
    return [p | data.get(p["name"], {}) for p in params]
