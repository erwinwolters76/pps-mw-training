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
    fill_value: float,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""

    all_data = xr.open_mfdataset(
        [f.decode("utf-8") for f in data_files],
        combine="nested",
        concat_dim="nscene",
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

    return [input_data, label_data]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
)
def load_data(matched_files, input_params, fill_value):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[matched_files, input_params, fill_value],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    matched_files: list[Path],
    batch_size: int,
    input_params: str,
    fill_value: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""
    ds = tf.data.Dataset.from_tensor_slices([f.as_posix() for f in matched_files])
    matched_files = [f.as_posix() for f in matched_files]
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x: load_data(
            x,
            tf.constant(input_params),
            tf.constant(fill_value),
        ),
    )
    return ds


def update_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calculate std and mean for the input dataset to normalise"""
    all_data = xr.open_mfdataset(
        [f for f in input_files],
        combine="nested",
        concat_dim="nscene",
    )

    input_data = np.stack(
        [all_data[p["name"]][:, :, :].values for idx, p in enumerate(input_params)],
        axis=3,
    )

    std = np.stack(
        [Scaler.get_std(input_data[:, :, :, idx]) for idx in range(input_data.shape[3])]
    )
    mean = np.stack(
        [
            Scaler.get_mean(input_data[:, :, :, idx])
            for idx in range(input_data.shape[3])
        ]
    )

    for idx, p in enumerate(input_params):
        p["std"] = std[idx]
    for idx, p in enumerate(input_params):
        p["mean"] = mean[idx]

    return input_params


def get_training_dataset(
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    input_params: list[dict[str, Any]],
    fill_value: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    # assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("*.nc*"))
    s = len(input_files)

    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)

    input_params = update_std_mean(
        input_files[0 : train_size + validation_size], input_params
    )
    params = json.dumps(input_params)

    # ds = _get_training_dataset(
    #     input_files,
    #     batch_size,
    #     params,
    #     fill_value,
    # )

    # return ds

    # train_size = int(s * train_fraction)
    # validation_size = int(s * validation_fraction)
    return [
        _get_training_dataset(
            f,
            batch_size,
            params,
            fill_value,
        )
        for f in [
            input_files[0:train_size],
            input_files[train_size : train_size + validation_size],
            input_files[train_size + validation_size : :],
        ]
    ]
