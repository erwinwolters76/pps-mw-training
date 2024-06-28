from pathlib import Path
from typing import Any, Optional
import datetime as dt
import json
import re

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr  # type: ignore


from pps_mw_training.utils.scaler import Scaler


def _load_data(
    data_files: np.ndarray,
    input_params: str,
    fill_value_input: float,
    fill_value_label: float,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""

    all_data = xr.open_mfdataset(
        [f.decode("utf-8") for f in data_files],
        combine="nested",
        concat_dim="nscene",
    )

    all_data = all_data.sel(
        {
            "npix": all_data["nscan"].values[8:-8],
            "nscan": all_data["npix"].values[8:-8],
        }
    ).load()

    input_params = json.loads(input_params)

    scaler = Scaler.from_dict(input_params)

    input_data = np.stack(
        [
            scaler.apply(all_data[p["name"]][:, :, :].values, idx)
            for idx, p in enumerate(input_params)
        ],
        axis=3,
    )

    label_data = all_data["cloud_base"][:, :, :].values
    input_data[~np.isfinite(input_data)] = fill_value_input
    label_data[~np.isfinite(label_data)] = fill_value_label
    label_data[label_data < 0] = fill_value_label
    label_data[label_data > 4500] = fill_value_label

    return [np.float32(input_data), np.float32(np.expand_dims(label_data, axis=3))]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
)
def load_data(matched_files, input_params, fill_value_input, fill_value_label):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[matched_files, input_params, fill_value_input, fill_value_label],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    matched_files: list[Path],
    batch_size: int,
    input_params: str,
    fill_value_input: float,
    fill_value_label: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    print(matched_files)
    ds = tf.data.Dataset.from_tensor_slices([f.as_posix() for f in matched_files])
    matched_files = [f.as_posix() for f in matched_files]
    ds = ds.batch(batch_size)

    ds = ds.map(
        lambda x: load_data(
            x,
            tf.constant(input_params),
            tf.constant(fill_value_input),
            tf.constant(fill_value_label),
        ),
    )
    return ds


def update_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calculate std and mean for the input dataset to normalise"""

    all_dict = {}
    with xr.open_dataset(input_files[0]) as ds:
        for key in ds.keys():
            all_dict[key] = {"n": 0, "x": 0, "x2": 0}

    for file in input_files:
        with xr.open_dataset(file) as ds:
            ds = ds.sel(
                {
                    "npix": ds["nscan"].values[8:-8],
                    "nscan": ds["npix"].values[8:-8],
                }
            )
            for key in ds.keys():
                all_dict[key]["n"] += (np.isfinite(ds[key].values)).size
                all_dict[key]["x"] += np.nansum(ds[key].values)
                all_dict[key]["x2"] += np.nansum(ds[key].values ** 2)

    for p in input_params:
        key = p["name"]
        p["std"] = np.sqrt(
            (all_dict[key]["x2"] / all_dict[key]["n"])
            - (all_dict[key]["x"] / all_dict[key]["n"]) ** 2
        )
        p["mean"] = all_dict[key]["x"] / all_dict[key]["n"]

    return input_params


def get_training_dataset(
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    input_params: list[dict[str, Any]],
    fill_value_input: float,
    fill_value_label: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    # assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("*.nc*"))[:250]
    s = len(input_files)

    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)

    input_params = update_std_mean(
        input_files[0 : train_size + validation_size], input_params
    )

    params = json.dumps(input_params)
    print(train_size, validation_size)
    return [
        _get_training_dataset(f, batch_size, params, fill_value_input, fill_value_label)
        for f in [
            input_files[0:train_size],
            input_files[train_size : train_size + validation_size],
            input_files[train_size + validation_size : :],
        ]
    ]
