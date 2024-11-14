from pathlib import Path
from typing import Any
import json
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr  # type: ignore
from pps_mw_training.utils.scaler import get_scaler


def _load_data(
    data_files: np.ndarray,
    input_parameters: str,
    label_parameters: str,
    fill_value_input: float,
    fill_value_label: float,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""
    with xr.open_mfdataset(
        [f.decode("utf-8") for f in data_files],
        combine="nested",
        concat_dim="nscene",
        engine="h5netcdf",
    ) as all_data:
        all_data = all_data.load()
        input_params = json.loads(input_parameters)
        label_params = json.loads(label_parameters)
        input_scaler = get_scaler(input_params)
        input_data = np.stack(
            [
                input_scaler.apply(all_data[p["name"]].values, idx)
                for idx, p in enumerate(input_params)
            ],
            axis=3,
        )
        label_scaler = get_scaler(label_params)
        label_data = np.stack(
            [
                label_scaler.apply(all_data[p["name"]].values, idx)
                for idx, p in enumerate(label_params)
            ],
            axis=3,
        )
        input_data[~np.isfinite(input_data)] = fill_value_input
        label_data[~np.isfinite(label_data)] = fill_value_label
        return [
            np.float32(input_data),
            np.float32(label_data),
        ]


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
)
def load_data(
    matched_files,
    input_params,
    label_params,
    fill_value_input,
    fill_value_label,
):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[
            matched_files,
            input_params,
            label_params,
            fill_value_input,
            fill_value_label,
        ],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    matched_files: list[Path],
    batch_size: int,
    input_params: str,
    label_params: str,
    fill_value_input: float,
    fill_value_label: float,
) -> tf.data.Dataset:
    """Get training dataset."""

    ds = tf.data.Dataset.from_tensor_slices(
        [f.as_posix() for f in matched_files]
    )
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x: load_data(
            x,
            tf.constant(input_params),
            tf.constant(label_params),
            tf.constant(fill_value_input),
            tf.constant(fill_value_label),
        ),
        num_parallel_calls=1,
    )
    return ds


def get_training_dataset(
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
    input_parameters: list[dict[str, Any]],
    label_parameters: list[dict[str, Any]],
    fill_value_input: float,
    fill_value_label: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("cnn_data*.nc*"))[:1000]

    s = len(input_files)
    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)

    input_params = json.dumps(input_parameters)
    label_params = json.dumps(label_parameters)
    return [
        _get_training_dataset(
            f,
            batch_size,
            input_params,
            label_params,
            fill_value_input,
            fill_value_label,
        )
        for f in [
            input_files[0:train_size],
            input_files[train_size : train_size + validation_size],
            input_files[train_size + validation_size :],
        ]
    ]
