from pathlib import Path
from typing import Any, Tuple
import json
import random
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr  # type: ignore
from scipy.ndimage import rotate  # type: ignore
from pps_mw_training.utils.scaler import Scaler
from pps_mw_training.utils.calculate_mean_std import get_std_mean


def _load_data(
    data_files: np.ndarray,
    input_params: str,
    fill_value_input: float,
    fill_value_label: float,
    training_label_name: bytes,
    training_label_max: float,
    training_label_min: float,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""

    with xr.open_mfdataset(
        [f.decode("utf-8") for f in data_files],
        combine="nested",
        concat_dim="nscene",
        engine="h5netcdf",
    ) as all_data:
        all_data = all_data.sel(
            {
                "npix": all_data["npix"].values,
                "nscan": all_data["nscan"].values,
            }
        ).load()
        params = json.loads(input_params)
        scaler = Scaler.from_dict(params)
        input_data = np.stack(
            [
                scaler.apply(all_data[p["name"]][:, :, :].values, idx)
                for idx, p in enumerate(params)
            ],
            axis=3,
        )
        label_data = all_data[training_label_name.decode("utf-8")].values
        input_data[~np.isfinite(input_data)] = fill_value_input
        label_data[~np.isfinite(label_data)] = fill_value_label
        label_data[label_data < training_label_min] = fill_value_label
        label_data[label_data >= training_label_max] = training_label_max
        label_data = (label_data - training_label_min) / (
            training_label_max - training_label_min
        )
        # extra augmentation as rotate
        # input_data, label_data = rotate_data(input_data, label_data)
        return [
            np.float32(input_data),
            np.float32(np.expand_dims(label_data, axis=3)),
        ]


def rotate_data(
    xtrain: np.ndarray, ytrain: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """rotate training data and label data
    should be in augmentation
    """
    xrotated = np.empty(xtrain.shape)
    yrotated = np.empty(ytrain.shape)
    batchsize = xtrain.shape[0]
    nchannels = xtrain.shape[-1]
    for i in range(batchsize):
        angle = random.choice([0.0, 90.0, 180.0, 270.0])
        yrotated[i, :, :] = rotate(ytrain[i, :, :], angle, reshape=False)
        for j in range(nchannels):
            xrotated[i, :, :, j] = rotate(
                xtrain[i, :, :, j], angle, reshape=False
            )
    return xrotated, yrotated


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=(None,), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )
)
def load_data(
    matched_files,
    input_params,
    fill_value_input,
    fill_value_label,
    training_label_name,
    training_label_max,
    training_label_min,
):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[
            matched_files,
            input_params,
            fill_value_input,
            fill_value_label,
            training_label_name,
            training_label_max,
            training_label_min,
        ],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    matched_files: list[Path],
    batch_size: int,
    input_params: str,
    fill_value_input: float,
    fill_value_label: float,
    training_label_name: str,
    training_label_max: float,
    training_label_min: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    ds = tf.data.Dataset.from_tensor_slices(
        [f.as_posix() for f in matched_files]
    )
    # matched_files = [f.as_posix() for f in matched_files]
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x: load_data(
            x,
            tf.constant(input_params),
            tf.constant(fill_value_input),
            tf.constant(fill_value_label),
            tf.constant(training_label_name),
            tf.constant(training_label_max),
            tf.constant(training_label_min),
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
    input_params: list[dict[str, Any]],
    training_label_name: str,
    training_label_max: float,
    training_label_min: float,
    fill_value_input: float,
    fill_value_label: float,
    update_std_mean: bool,
) -> list[tf.data.Dataset]:
    """Get training dataset."""

    assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("cnn_data*.nc*"))[:1000]

    s = len(input_files)
    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)

    if update_std_mean:
        input_params = get_std_mean(
            input_files[0: train_size + validation_size], input_params
        )
    params = json.dumps(input_params)
    return [
        _get_training_dataset(
            f,
            batch_size,
            params,
            fill_value_input,
            fill_value_label,
            training_label_name,
            training_label_max,
            training_label_min,
        )
        for f in [
            input_files[0:train_size],
            input_files[train_size: train_size + validation_size],
            input_files[train_size + validation_size:],
        ]
    ]
