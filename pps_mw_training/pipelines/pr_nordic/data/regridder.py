from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence, Union
import logging


from pyproj import Transformer  # type: ignore
from scipy.interpolate import griddata  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


WGS84 = "EPSG:4326"
DELTA_GEO = 100e3  # [m]
DELTA_TIME = 1200 * 1e9  # [ns]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Regridder:
    """Class for regridding of data onto a new grid"""

    dataset: xr.Dataset
    grid: xr.Dataset

    def regrid(
        self,
        channels: Sequence[Enum],
        method: str = "linear",
    ) -> Optional[xr.Dataset]:
        """Regrid data onto the given grid for all channels."""
        datasets = [self._regrid(channel, method) for channel in channels]
        if not any([d.attrs["valid_data"] for d in datasets]):
            logging.warning("Found no valid data within grid.")
            return None
        return xr.merge(datasets)

    def transform(
        self,
        data: xr.DataArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform data coordinates to grid coordinates."""
        transformer = Transformer.from_crs(WGS84, self.grid.attrs["crs"])
        return transformer.transform(data.lat, data.lon)

    def _get_attrs(
        self,
        time: datetime,
        valid_data: int,
    ) -> dict[str, Union[str, int]]:
        """Get attributes."""
        attrs = {
            "platform": self.dataset.attrs["platform"],
            "sensor": self.dataset.attrs["sensor"],
            "time": time,
            "valid_data": valid_data,
        }
        return self.grid.attrs | attrs

    def _within_grid_limits(
        self,
        x: np.ndarray,
        y: np.ndarray,
        delta: float = DELTA_GEO,
    ) -> np.ndarray:
        """Filter data on geolocation."""
        return (
            (y >= self.grid.ys.values.min() - delta)
            & (y <= self.grid.ys.values.max() + delta)
            & (x >= self.grid.xs.values.min() - delta)
            & (x <= self.grid.xs.values.max() + delta)
        )

    @staticmethod
    def _within_time_limits(
        timestamps: np.ndarray,
        t0: float,
        delta: float = DELTA_TIME,
    ) -> np.ndarray:
        """Filter data on time."""
        return (timestamps >= t0 - delta) & (timestamps <= t0 + delta)

    @staticmethod
    def _get_timestamps(
        data: xr.DataArray,
    ) -> np.ndarray:
        """Get timestamps."""
        timestamps = np.array([t.astype(datetime) for t in data.time])
        timestamps[timestamps == None] = 0.  # noqa: E711
        return timestamps

    @staticmethod
    def _get_median_timestamp(
        data: xr.DataArray,
        timestamps: np.ndarray,
        filt_geo: np.ndarray
    ) -> float:
        """Get median timestamp of the geo filtered data."""
        idxs = np.array([
            idx for idx in np.arange(data.time.values.size)
            if np.any(filt_geo[idx])
        ])
        return np.median(timestamps[idxs])

    @staticmethod
    def _merge_filt(
        filt_geo: np.ndarray,
        filt_time: np.ndarray,
    ) -> np.ndarray:
        """Merge geo and time filters."""
        for idx in np.arange(filt_time.size):
            if not filt_time[idx]:
                filt_geo[idx] = False
        return filt_geo

    def _get_filt(
        self,
        x: np.ndarray,
        y: np.ndarray,
        data: xr.DataArray,
    ) -> tuple[np.ndarray, datetime]:
        """Get a gelocation and time filter."""
        within_grid = self._within_grid_limits(x, y)
        if not within_grid.any():
            raise ValueError("No data within grid")
        # Filter data on time such that data from both ends of an
        # orbit is not included
        timestamps = self._get_timestamps(data)
        timestamp = self._get_median_timestamp(data, timestamps, within_grid)
        within_time = self._within_time_limits(timestamps, timestamp)
        filt = self._merge_filt(within_grid, within_time)
        return filt, datetime.utcfromtimestamp(timestamp / 1e9)

    def _regrid(
        self,
        channel: Enum,
        method: str,
    ) -> xr.Dataset:
        """Regrid data onto the given grid for a single channel."""
        dataset = self.dataset[channel]
        x, y = self.transform(dataset)
        filt, time = self._get_filt(x, y, dataset)
        data = griddata(
            np.array([y[filt], x[filt]]).transpose(),
            dataset.values.squeeze()[filt],
            (self.grid.ys.values, self.grid.xs.values),
            method=method,
        )
        s1, s2 = data.shape
        valid_data = 1
        if not np.isfinite(data).any():
            valid_data = 0
            logging.warning(
                f"No finite data in bbox for channel {channel}."
            )
        if not np.isfinite(data[int(s1 / 2), int(s2 / 2)]):
            valid_data = 0
            logging.warning(
                f"Low coverage in bbox for channel {channel}."
            )
        return xr.Dataset(
            data_vars={channel: (("y", "x"), data)},
            attrs=self._get_attrs(time, valid_data),
        )
