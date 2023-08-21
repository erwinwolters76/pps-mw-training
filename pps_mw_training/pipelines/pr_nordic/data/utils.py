from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from pps_mw_training.pipelines.pr_nordic.data.data_model import BANDS


FILENAME_FMT = "mw_%Y%m%d_%H_%M.nc"
MINUTE = 15
COMPRESSION = {
    "dtype": "int16",
    "scale_factor": 0.01,
    "zlib": True,
    "_FillValue": -99,
}


@dataclass
class Writer:
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

    @staticmethod
    def fix_attrs(
        attrs: dict[str, Any]
    ) -> dict[str, Any]:
        """Fix attributess."""
        if "time" in attrs and isinstance(attrs["time"], datetime):
            attrs["time"] = attrs["time"].isoformat()
        if "crs" in attrs:
            attrs["crs"] = str(attrs["crs"])
        return attrs

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
            attrs=self.fix_attrs(self.dataset.attrs),
        )

    def get_reshaped_dataset(self) -> xr.Dataset:
        """Get reshaped dataset."""
        data = self._get_empty_band_dataset()
        for c in self.dataset:
            for b in BANDS:
                for idx, names in BANDS[b].items():
                    if c.name in names:
                        data[b].values[:, :, idx] = self.dataset[c].values
        return data

    def write(
        self,
    ) -> Path:
        """Write data to file."""
        outfile = self.output_filepath
        dataset = self.get_reshaped_dataset()
        dataset.to_netcdf(
            outfile,
            encoding={band: COMPRESSION for band in dataset.variables},
        )
        return outfile


def reshape_filelist(
    files: list[Path],
    chunk_size: int,
) -> list[list[Path]]:
    """Reshape list of files to a list of list of files."""
    return [
        files[idx * chunk_size: idx * chunk_size + chunk_size]
        for idx in range(ceil(len(files) / chunk_size))
    ]
