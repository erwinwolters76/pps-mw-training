from pathlib import Path
import json
from typing import Optional
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
    ) as all_data:
        return [
            scale_data(all_data, input_parameters, fill_value_input),
            scale_data(all_data, label_parameters, fill_value_label),
        ]


def scale_data(
    data: xr.Dataset, parameters: str, fill_value: float
) -> np.ndarray:
    """Scale data"""
    params = json.loads(parameters)
    scaler = get_scaler(params)
    data = np.stack(
        [
            np.where(
                scaler.apply(
                    data[p["name"]].values == fill_value,
                    np.nan,
                    data[p["name"]].values,
                ),
                idx,
            )
            for idx, p in enumerate(params)
        ],
        axis=3,
        dtype=np.float32,
    )
    data[~np.isfinite(data)] = fill_value
    return data


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
    files,
    input_params,
    label_params,
    fill_value_input,
    fill_value_label,
):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[
            files,
            input_params,
            label_params,
            fill_value_input,
            fill_value_label,
        ],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    files: list[Path],
    batch_size: int,
    input_parameters: list[dict[str, str | float]],
    label_parameters: list[dict[str, str | float]],
    fill_value_input: float,
    fill_value_label: float,
) -> tf.data.Dataset:
    """Get training dataset."""

    ds = tf.data.Dataset.from_tensor_slices([f.as_posix() for f in files])
    ds = ds.batch(batch_size)
    input_params = json.dumps(input_parameters)
    label_params = json.dumps(label_parameters)
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
    input_parameters: list[dict[str, str | float]],
    label_parameters: list[dict[str, str | float]],
    fill_value_input: float,
    fill_value_label: float,
    file_limit: Optional[int] = None,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("cnn_data*.nc"))
    if file_limit:
        input_files = input_files[:file_limit]

    s = len(input_files)
    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)

    return [
        _get_training_dataset(
            f,
            batch_size,
            input_parameters,
            label_parameters,
            fill_value_input,
            fill_value_label,
        )
        for f in [
            input_files[0:train_size],
            input_files[train_size : train_size + validation_size],
            input_files[train_size + validation_size :],
        ]
    ]
