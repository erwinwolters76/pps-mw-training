import argparse
from typing import Any
from pathlib import Path
import json

import numpy as np  # type: ignore
import xarray as xr  # type: ignore
from pps_mw_training.pipelines.cloud_base.settings import INPUT_PARAMS


def get_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Calculate standard deviation and mean for the input dataset to normalize.
    """
    stats = get_stats(input_files, input_params)

    for p in input_params:
        parameter = p["name"]
        mean, std = calculate_mean_std(stats[parameter])
        p["mean"] = mean
        p["std"] = std

    return input_params


def get_stats(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Calculate intermediate statistics to calculate mean and std
    """
    stats = {}
    for idx, file in enumerate(input_files):
        with xr.open_dataset(file.as_posix()) as ds:
            for p in input_params:
                parameter = p["name"]
                data = ds[parameter].values
                if idx == 0:
                    stats[parameter] = {
                        "n": np.count_nonzero(np.isfinite(data)),
                        "sum": np.nansum(data),
                        "sum_of_squares": np.nansum(data**2),
                    }
                    continue
                stats[parameter]["n"] += np.count_nonzero(np.isfinite(data))
                stats[parameter]["sum"] += np.nansum(data)
                stats[parameter]["sum_of_squares"] += np.nansum(data**2)
    return stats


def calculate_mean_std(stats: dict[str, float]) -> tuple[float, float]:
    """
    Calculate mean and standard deviation from intermediate stats
    """
    mean = stats["sum"] / stats["n"]
    std = np.sqrt((stats["sum_of_squares"] / stats["n"]) - (mean**2))
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Calculate std and mean for input datasets."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory containing input dataset files.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    input_files = list((input_dir).glob("cnn_data*.nc*"))[:100]
    input_params = get_std_mean(input_files, INPUT_PARAMS)
    with open("input_params.json", "w") as outfile:
        outfile.write(json.dumps(input_params, indent=4))


if __name__ == "__main__":
    main()
