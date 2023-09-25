from pathlib import Path
from typing import Any, Optional
import datetime as dt
import json
import re

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import xarray as xr  # type: ignore


from pps_mw_training.utils.scaler import Scaler


def get_file_info(
    data_file: Path,
) -> Optional[dt.datetime]:
    """Get time from file name."""
    m = re.match(
        r"[a-z]+_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})_(?P<minute>\d{2})",  # noqa: E501
        data_file.stem,
    )
    if m is not None:
        d = m.groupdict()
        return dt.datetime.fromisoformat(
            f'{d["year"]}-{d["month"]}-{d["day"]}T{d["hour"]}:{d["minute"]}'
        )
    return None


def match_files(
    satellite_files: list[Path],
    radar_files: list[Path],
) -> list[tuple[Path, Path]]:
    """Get matched files."""
    matched_files: list[tuple[Path, Path]] = []
    radar_file_info = [get_file_info(f) for f in radar_files]
    for satellite_file in satellite_files:
        satellite_file_info = get_file_info(satellite_file)
        if satellite_file_info is None:
            continue
        try:
            index = radar_file_info.index(satellite_file_info)
        except ValueError:
            continue
        matched_files.append((satellite_file, radar_files[index]))
    return matched_files


def _load_data(
    mw_files: np.ndarray,
    radar_files: np.ndarray,
    input_parameters: str,
    qi_min: float,
    distance_max: float,
    fill_value_mw: float,
    fill_value_radar: float,
    n: int = 16,
    res: int = 2,
) -> list[np.ndarray]:
    """Load, scale, and filter data."""
    mw_data = xr.open_mfdataset(
        [f.decode("utf-8") for f in mw_files],
        combine="nested",
        concat_dim="time",
    )
    radar_data = xr.open_mfdataset(
        [f.decode("utf-8") for f in radar_files],
        combine="nested",
        concat_dim="time",
    )
    x = n * (mw_data.x.size // n)
    y = n * (mw_data.y.size // n)
    n = radar_data.y.size // mw_data.y.size
    mw_data = mw_data.sel(
        {
            "y": mw_data["y"].values[0: y],
            "x": mw_data["x"].values[0: x],
        }
    ).load()
    radar_data = radar_data.sel(
        {
            "y": radar_data["y"].values[0: n * y: res],
            "x": radar_data["x"].values[0: n * x: res],
        }
    ).load()
    input_params = json.loads(input_parameters)
    scaler = Scaler.from_dict(input_params)
    mw_data = np.stack(
        [
            scaler.apply(
                mw_data[p["band"]][:, :, :, p["index"]].values,
                idx,
            ) for idx, p in enumerate(input_params)
        ],
        axis=3,
    )
    mw_data[~np.isfinite(mw_data)] = fill_value_mw
    radar_data.dbz.values[
        ~(
            (radar_data.qi >= qi_min)
            & (radar_data.distance_radar <= distance_max)
        )
    ] = fill_value_radar
    radar_data.dbz.values[
        ~np.isfinite(radar_data.dbz.values)
    ] = fill_value_radar
    return [
        mw_data,
        np.expand_dims(radar_data.dbz.values, axis=3),
    ]


@tf.function(input_signature=(
    tf.TensorSpec(shape=(None,), dtype=tf.string),
    tf.TensorSpec(shape=(None,), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.string),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32),
))
def load_data(
    mw_files,
    radar_files,
    input_params,
    qi_min,
    distance_max,
    fill_value_mw,
    fill_value_radar,
):
    """Load netcdf dataset."""
    return tf.numpy_function(
        func=_load_data,
        inp=[
            mw_files,
            radar_files,
            input_params,
            qi_min,
            distance_max,
            fill_value_mw,
            fill_value_radar,
        ],
        Tout=[tf.float32, tf.float32],
    )


def _get_training_dataset(
    files: list[tuple[Path, Path]],
    batch_size: int,
    qi_min: float,
    distance_max: float,
    input_params: str,
    fill_value_mw: float,
    fill_value_radar: float,
) -> tf.data.Dataset:
    """Get training dataset."""
    ds = tf.data.Dataset.from_tensor_slices(
        (
            [f.as_posix() for f, _ in files],
            [f.as_posix() for _, f in files],
        )
    )
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda x, y: load_data(
            x,
            y,
            tf.constant(input_params),
            tf.constant(qi_min),
            tf.constant(distance_max),
            tf.constant(fill_value_mw),
            tf.constant(fill_value_radar),
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
    qi_min: float,
    distance_max: float,
    input_params: list[dict[str, Any]],
    fill_value_mw: float,
    fill_value_radar: float,
) -> list[tf.data.Dataset]:
    """Get training dataset."""
    params = json.dumps(input_params)
    sat_files = list((training_data_path / "satellite").glob('*.nc*'))
    radar_files = list((training_data_path / "radar").glob('*.nc*'))
    files = match_files(sat_files, radar_files)
    s = len(files)
    assert train_fraction + validation_fraction + test_fraction == 1
    train_size = int(s * train_fraction)
    validation_size = int(s * validation_fraction)
    return [
        _get_training_dataset(
            f,
            batch_size,
            qi_min,
            distance_max,
            params,
            fill_value_mw,
            fill_value_radar,
        ) for f in [
            files[0: train_size],
            files[train_size: train_size + validation_size],
            files[train_size + validation_size::],
        ]
    ]
