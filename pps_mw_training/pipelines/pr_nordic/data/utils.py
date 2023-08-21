from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_training.pipelines.pr_nordic.data.data_model import (
    BANDS, ChannelAtms
)


FILENAME_FMT = "mw_%Y%m%d_%H_%M.nc"
MINUTE = 15
COMPRESSION = {
    "dtype": "int16",
    "scale_factor": 0.01,
    "zlib": True,
    "_FillValue": -99,
}


@dataclass
class Reshaper:
    """Reshaper class."""

    dataset: xr.Dataset

    def _get_empty_band_dataset(self) -> xr.Dataset:
        """Get empty band dataset."""
        return xr.Dataset(
            {
                band: (
                    ("y", "x", f"channel_{band}"),
                    np.full(
                        (
                            self.dataset.y.size,
                            self.dataset.x.size,
                            len(BANDS[band]),
                        ),
                        np.nan,
                    ),
                ) for band in BANDS
            },
        )

    @property
    def attrs(self) -> dict[str, Any]:
        """Fix attributess."""
        attrs = self.dataset.attrs
        if "time" in attrs and isinstance(attrs["time"], datetime):
            attrs["time"] = attrs["time"].isoformat()
        if "crs" in attrs:
            attrs["crs"] = str(attrs["crs"])
        return attrs

    def reshape(self) -> xr.Dataset:
        """Get dataset as a band dataset."""
        data = self._get_empty_band_dataset()
        for c in self.dataset:
            for b in BANDS:
                for idx, names in BANDS[b].items():
                    if c.name in names:
                        data[b].values[:, :, idx] = self.dataset[c].values
        data.attrs = self.attrs
        return data


@dataclass
class Writer(Reshaper):
    """Writer class."""

    dataset: xr.Dataset
    output_path: Path

    @property
    def output_filepath(
        self,
    ) -> Path:
        """Get filepath."""
        output_path = self.output_path / "satellite"
        output_path.mkdir(parents=True, exist_ok=True)
        t0 = self.round_time(self.dataset.attrs["time"])
        return output_path / t0.strftime(FILENAME_FMT)

    @staticmethod
    def round_time(
        t0: datetime,
        minute: int = MINUTE,
    ) -> datetime:
        """Round time to e.g. nearest quarter hour by setting minute to 15."""
        return (
            datetime(t0.year, t0.month, t0.day, t0.hour)
            + timedelta(minutes=(minute * round(t0.minute / minute)))
        )

    def write(
        self,
    ) -> Path:
        """Write data to file."""
        output_filepath = self.output_filepath
        dataset = self.reshape()
        dataset.to_netcdf(
            output_filepath,
            encoding={band: COMPRESSION for band in dataset.variables},
        )
        return output_filepath


def reshape_filelist(
    files: list[Path],
    chunk_size: int,
) -> list[list[Path]]:
    """Reshape list of files to a list of list of files."""
    return [
        files[idx * chunk_size: idx * chunk_size + chunk_size]
        for idx in range(ceil(len(files) / chunk_size))
    ]


def get_channels(
    params: list[dict[str, Any]],
) -> Sequence[Enum]:
    """Get sequence of channels from params dict."""
    return [ChannelAtms[BANDS[p["band"]][p["index"]][0]] for p in params]
