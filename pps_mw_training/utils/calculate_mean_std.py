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
    """calculate std and mean for the input dataset to normalise"""

    stats = {}
    for idx, file in enumerate(input_files):
        with xr.open_dataset(file.as_posix()) as ds:
            for parameter in ds:
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

    for p in input_params:
        parameter = p["name"]
        p["mean"] = stats[parameter]["sum"] / stats[parameter]["n"]
        p["std"] = np.sqrt(
            (stats[parameter]["sum_of_squares"] / stats[parameter]["n"])
            - (stats[parameter]["sum"] / stats[parameter]["n"]) ** 2
        )

    return input_params


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
    input_files = list((input_dir).glob("cnn_data*.nc*"))
    input_params = get_std_mean(input_files, INPUT_PARAMS)
    with open("input_params.json", "w") as outfile:
        outfile.write(json.dumps(input_params, indent=4))


if __name__ == "__main__":
    main()
