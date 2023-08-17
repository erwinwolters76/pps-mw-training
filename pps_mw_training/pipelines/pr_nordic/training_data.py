from pathlib import Path
from typing import Optional
import datetime as dt
import re

import numpy as np  # type: ignore
import xarray as xr  # type: ignore


BANDS = {
    "mw_low": {
        0: ["ATMS_01", "AMSUA_01"],  # 23.8 GHz
        1: ["ATMS_02", "AMSUA_02"],  # 31.4 GHz
    },
    "mw_50": {
        0: ["ATMS_03", "AMSUA_03"],  # 50.3 GHz
        1: ["ATMS_04"],  # 51.76 GHz
        2: ["ATMS_05", "AMSUA_04"],  # 52.8 GHz
        3: ["ATMS_06", "AMSUA_05"],  # 53.596 +/- 0.115
        4: ["ATMS_07", "AMSUA_06"],  # 54.4 GHz
        5: ["ATMS_08", "AMSUA_07"],  # 54.94 GHz
        6: ["ATMS_09", "AMSUA_08"],  # 55.5 GHz
        7: ["ATMS_10", "AMSUA_09"],  # 57.290344 GHz
        8: ["ATMS_11", "AMSUA_10"],  # 57.290344 +/- 0.217 GHz
        9: ["ATMS_12", "AMSUA_11"],  # 57.290344 +/- 0.3222 +/- 0.048 GHz
        10: ["ATMS_13", "AMSUA_12"],  # 57.290344 +/- 0.3222 +/- 0.022 GHz
        11: ["ATMS_14", "AMSUA_13"],  # 57.290344 +/- 0.3222 +/- 0.010 GHz
        12: ["ATMS_15", "AMSUA_14"],  # 57.290344 +/- 0.3222 +/- 0.0045 GHz
    },
    "mw_90": {
        0: ["ATMS_16"],  # 88.2 GHz
        1: ["MHS_01", "AMSUA_15"],  # 89 GHz
    },
    "mw_160": {
        0: ["ATMS_17"],  # 166 GHz
        1: ["MHS_02"],  # 157 GHz
    },
    "mw_183": {
        0: ["ATMS_18", "MHS_05"],  # 183 +/- 1 GHz
        1: ["ATMS_19"],  # 183 +/- 1.8 GHz
        2: ["ATMS_20", "MHS_04"],  # 183 +/- 3, GHz
        3: ["ATMS_21"],  # 183 +/- 4.5 GHz
        4: ["ATMS_22", "MHS_03"],  # 183 +/- 7 GHz
    },
}
TRAINING_DATA_SHAPE = tuple[xr.Dataset, xr.DataArray]


def load_netcdf_data(
    netcdf_file: Path,
) -> xr.Dataset:
    """Load netcdf dataset."""
    with xr.open_dataset(netcdf_file) as ds:
        return ds


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
        index = radar_file_info.index(satellite_file_info)
        matched_files.append((satellite_file, radar_files[index]))
    return matched_files


def filter_radar_data(
    data: xr.Dataset,
    qi_min: float,
    distance_max: float,
) -> xr.Dataset:
    """Filter radar data on quality and distance from radar."""
    filt = (
        (data.qi >= qi_min)
        & (data.distance_radar <= distance_max)
    )
    data.dbz.values[~filt.values] = np.nan
    return data


def split_training_data(
    mw_data: xr.Dataset,
    radar_data: xr.Dataset,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[TRAINING_DATA_SHAPE, TRAINING_DATA_SHAPE, TRAINING_DATA_SHAPE]:
    """Split training data into three parts."""
    n_samples = mw_data.time.size
    fractions = [train_fraction, validation_fraction, test_fraction]
    limits = np.cumsum([int(f * n_samples) for f in fractions])
    idxs = np.random.permutation(n_samples)
    idxs_train = idxs[0: limits[0]]
    idxs_val = idxs[limits[0]: limits[1]]
    idxs_test = idxs[limits[1]: limits[2]]
    return (
        (
            mw_data.sel({"time": idxs_train}),
            radar_data.sel({"time": idxs_train}).dbz,
        ),
        (
            mw_data.sel({"time": idxs_val}),
            radar_data.sel({"time": idxs_val}).dbz,
        ),
        (
            mw_data.sel({"time": idxs_test}),
            radar_data.sel({"time": idxs_test}).dbz,
        ),
    )


def fix_data_shape(
    mw_data: xr.Dataset,
    radar_data: xr.Dataset,
    x: int = 224,
    y: int = 272,
    res: int = 2,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Fix the shape of the training dataset."""
    n = radar_data.y.size // mw_data.y.size
    mw_data = mw_data.sel(
        {
            "y": mw_data["y"].values[0: y],
            "x": mw_data["x"].values[0: x],
        }
    )
    radar_data = radar_data.sel(
        {
            "y": radar_data["y"].values[0: n * y: res],
            "x": radar_data["x"].values[0: n * x: res],
        }
    )
    return mw_data, radar_data


def get_training_dataset(
    training_data_path: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    qi_min: float,
    distance_max: float,
) -> tuple[TRAINING_DATA_SHAPE, TRAINING_DATA_SHAPE, TRAINING_DATA_SHAPE]:
    """Get training dataset."""
    mw_files = list((training_data_path / "microwave").glob('*.nc*'))
    radar_files = list((training_data_path / "radar").glob('*.nc*'))
    files = match_files(mw_files, radar_files)
    mw_data = xr.concat([load_netcdf_data(f) for f, _ in files], dim="time")
    radar_data = xr.concat([load_netcdf_data(f) for _, f in files], dim="time")
    mw_data, radar_data = fix_data_shape(mw_data, radar_data)
    radar_data = filter_radar_data(radar_data, qi_min, distance_max)
    return split_training_data(
        mw_data,
        radar_data,
        train_fraction,
        validation_fraction,
        test_fraction,
    )
