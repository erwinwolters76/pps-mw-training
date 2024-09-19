from pathlib import Path
from typing import Any, Optional
import datetime as dt
import json
import re
import random
import pickle
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr  # type: ignore
from scipy.ndimage import rotate

from pps_mw_training.utils.scaler import Scaler


def _load_data(
    data_files: np.ndarray,
    input_params: str,
    fill_value_input: float,
    fill_value_label: float,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""

    with  xr.open_mfdataset(
        [f.decode("utf-8") for f in data_files],
        #[f for f in data_files],
        combine="nested",
        concat_dim="nscene",
    ) as all_data:

        all_data = all_data.sel(
            {
                "npix": all_data["npix"].values[8:-8],
                "nscan": all_data["nscan"].values[8:-8],
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
        #label_data = all_data["cloud_top"][:, :, :].values
        input_data[~np.isfinite(input_data)] = fill_value_input
        label_data[~np.isfinite(label_data)] = fill_value_label
        label_data[label_data < 0] = fill_value_label
        label_data[label_data >= 12000] = 12000.0

        label_data[label_data > 0]  = (label_data[label_data > 0] - 0.)/(12000. - 0.)
        input_data, label_data = rotate_data(input_data, label_data)

        return [np.float32(input_data), np.float32(np.expand_dims(label_data, axis=3))]


def rotate_data(xtrain, ytrain):
    xrotated = np.empty(xtrain.shape)
    yrotated = np.empty(ytrain.shape)
    batchsize = xtrain.shape[0]
    nchannels = xtrain.shape[-1]
    for i in range(batchsize):
        angle = random.choice([0., 90., 180., 270.])
        yrotated[i, :, :] = rotate(ytrain[i, :, :], angle, reshape=False)
        for j in range(nchannels):
            xrotated[i, :, :, j] = rotate(xtrain[i, :, :, j], angle, reshape=False)
    return xrotated, yrotated


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
                all_dict[key]["n"] += np.sum((np.isfinite(ds[key].values)))
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
    print(training_data_path)
    # assert train_fraction + validation_fraction + test_fraction == 1
    input_files = list((training_data_path).glob("cnn_data*.nc*"))[:22000]

    s = len(input_files)
    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)
    # input_params = update_std_mean(
    #     input_files[0 : train_size + validation_size], input_params
    # )
    with open("/home/sm_indka/pps-mw-training/scripts/input_params.pickle", "rb") as f:
        input_params_all = pickle.load(f)

    # Create a dictionary from list2 for easy lookup by name
    dict2 = {item['name']: item for item in input_params_all}

    # Create the new list by merging the relevant fields
    new_list = []
    for item in input_params:
        name = item['name']
        if name in dict2:
            # Create a new dictionary combining fields from list1 and std, mean from list2
            new_item = {
                "name": item["name"],
                "scale": item["scale"],
                "min": item["min"],
                "max": item["max"],
                "zscore_normalise": item["zscore_normalise"],
                "std": dict2[name]["std"],
                "mean": dict2[name]["mean"]
            }
            new_list.append(new_item)

    params = json.dumps(new_list)
    return [
        _get_training_dataset(f, batch_size, params, fill_value_input, fill_value_label)
        for f in [
            input_files[0:train_size],
            input_files[train_size : train_size + validation_size],
            input_files[train_size + validation_size : :],
        ]
    ]
