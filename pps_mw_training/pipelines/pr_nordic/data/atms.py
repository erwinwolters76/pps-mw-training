from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Union

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_training.pipelines.pr_nordic.data.data_model import (
    ChannelAtms, Platform,
)


@dataclass
class AtmsL1bReader:
    """Class for reading atms l1b file.

    This class handles the reading of ATMS data available for
    both JPSS1 (NOAA-20) and Suomi-NPP at:
    https://sounder.gesdisc.eosdis.nasa.gov/data/
    """

    l1b_file: Path

    @cached_property
    def _data(self) -> xr.Dataset:
        with xr.open_dataset(self.l1b_file) as ds:
            return ds

    @property
    def attrs(self) -> Dict[str, Union[str, Platform]]:
        platform = Platform.from_string(self._data.attrs["platform"])
        return self._data.attrs | {"platform": platform, "sensor": "ATMS"}

    @property
    def tb(self) -> np.ndarray:
        return np.moveaxis(self._data.antenna_temp.values, -1, 0)

    @property
    def time(self) -> np.ndarray:
        return self._data.obs_time_tai93.values[:, 0]

    @property
    def lat(self) -> np.ndarray:
        return self._data.lat.values

    @property
    def lon(self) -> np.ndarray:
        return self._data.lon.values

    @property
    def zenith_angle(self) -> np.ndarray:
        return self._data.sat_zen.values

    def get_data(self) -> xr.Dataset:
        """Get level1b dataset."""
        return xr.Dataset(
            data_vars={
                "zenith_angle": (("y", "x"), self.zenith_angle),
                "tb": (("channel", "y", "x"), self.tb),
            },
            coords={
                "time": ("y", self.time),
                "lat": (("y", "x"), self.lat),
                "lon": (("y", "x"), self.lon),
                "channel": ("channel", list(ChannelAtms)),
            },
            attrs=self.attrs,
        )
