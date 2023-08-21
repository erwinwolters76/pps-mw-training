from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Union

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
        return self._data.attrs | {"platform": platform.name, "sensor": "ATMS"}

    def get_ta_dataset(self) -> xr.Dataset:
        """Get antenna temperature dataset."""
        return xr.Dataset(
            data_vars={
                channel: (
                    ("y", "x"),
                    self._data.antenna_temp.values[:, :, channel.value],
                )
                for channel in ChannelAtms
            }
        )

    def get_geolocation_dataset(self) -> xr.Dataset:
        """Get geolocation dataset."""
        return xr.Dataset(
            data_vars={
                "sun_zenith": (("y", "x"), self._data.sol_zen.values),
                "sun_azimuth": (("y", "x"), self._data.sol_azi.values),
                "sat_zenith": (("y", "x"), self._data.sat_zen.values),
                "sat_azimuth": (("y", "x"), self._data.sat_azi.values),
            },
            coords={
                "time": ("y", self._data.obs_time_tai93.values[:, 0]),
                "lat": (("y", "x"), self._data.lat.values),
                "lon": (("y", "x"), self._data.lon.values),
            },
            attrs=self.attrs,
        )

    def _get_data(self) -> xr.Dataset:
        """Get level1b dataset."""
        return xr.merge(
            [self.get_geolocation_dataset(), self.get_ta_dataset()]
        )

    @classmethod
    def get_data(
        cls,
        l1b_files: list[Path],
    ) -> xr.Dataset:
        """Get ATMS level1b data from a list of level1b files."""
        return xr.concat([cls(f)._get_data() for f in l1b_files], dim="y")
