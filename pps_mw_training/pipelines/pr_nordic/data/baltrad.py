from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Tuple

from pyproj import CRS, Transformer  # type: ignore
import h5py  # type: ignore
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


from pps_mw_training.pipelines.pr_nordic.data.regridder import WGS84


BALTRAD_FILE = "comp_pcappi_blt2km_pn150_{datestr}_0x40000000001.h5"


@dataclass
class BaltradReader:
    """Class for reading baltrad file."""

    product_file: Path

    @classmethod
    def from_datetime(
        cls,
        t0: datetime,
        basepath: Path,
    ) -> "BaltradReader":
        return cls(
            basepath
            / t0.strftime("%Y/%m/%d/%H/%M")
            / BALTRAD_FILE.format(datestr=t0.strftime("%Y%m%dT%H%M%SZ"))
        )

    @cached_property
    def _data(self):
        return h5py.File(self.product_file, "r")

    @property
    def crs(self) -> CRS:
        projdef = self._data['where'].attrs.get("projdef").decode('ascii')
        return CRS(projdef)

    @property
    def lower_left(self) -> Tuple[float, float]:
        return (
            self._data['where'].attrs.get("LL_lat"),
            self._data['where'].attrs.get("LL_lon"),
        )

    @property
    def upper_left(self) -> Tuple[float, float]:
        return (
            self._data['where'].attrs.get("UL_lat"),
            self._data['where'].attrs.get("UL_lon"),
        )

    @property
    def lower_right(self) -> Tuple[float, float]:
        return (
            self._data['where'].attrs.get("LR_lat"),
            self._data['where'].attrs.get("LR_lon"),
        )

    @property
    def x_size(self) -> int:
        return self._data['where'].attrs.get("xsize")

    @property
    def y_size(self) -> int:
        return self._data['where'].attrs.get("ysize")

    @property
    def x_scale(self) -> int:
        return self._data['where'].attrs.get("xscale")

    @property
    def y_scale(self) -> int:
        return self._data['where'].attrs.get("yscale")

    @property
    def x(self) -> np.ndarray:
        transformer = Transformer.from_crs(WGS84, self.crs)
        left, _ = transformer.transform(*self.lower_left)
        return np.linspace(
            left,
            left + self.x_scale * (self.x_size - 1),
            self.x_size,
        )

    @property
    def y(self) -> np.ndarray:
        transformer = Transformer.from_crs(WGS84, self.crs)
        _, lower = transformer.transform(*self.lower_left)
        return np.linspace(
            lower + self.y_scale * (self.y_size - 1),
            lower,
            self.y_size,
        )

    @property
    def dbz(self) -> xr.DataArray:
        dbz = self._data["dataset1"]["data3"]
        what = dbz["what"]
        return xr.DataArray(
            data=dbz["data"][:],
            dims=("y", "x"),
            attrs={
                "scale_factor": what.attrs.get("gain"),
                "add_offset": what.attrs.get("offset"),
                "missing_value": what.attrs.get("nodata"),
                "undetect": what.attrs.get("undetect"),
                "quantity": what.attrs.get("quantity").decode(),
            },
        )

    @property
    def quality_index(self) -> xr.DataArray:
        qi = self._data["dataset1"]["data3"]["quality4"]
        what = qi["what"]
        return xr.DataArray(
            data=qi["data"][:],
            dims=("y", "x"),
            attrs={
                "scale_factor": what.attrs.get("gain"),
                "add_offset": what.attrs.get("offset"),
            },
        )

    @property
    def distance_radar(self) -> xr.DataArray:
        distance = self._data["dataset1"]["data3"]["quality3"]
        what = distance["what"]
        return xr.DataArray(
            data=distance["data"][:],
            dims=("y", "x"),
            attrs={
                "scale_factor": what.attrs.get("gain"),
                "add_offset": what.attrs.get("offset"),
            },
        )

    def get_data(self) -> xr.Dataset:
        x, y = np.meshgrid(self.x, self.y)
        return xr.Dataset(
            data_vars={
                "dbz": self.dbz,
                "qi": self.quality_index,
                "distance_radar": self.distance_radar,
            },
            attrs={"crs": self.crs},
        )

    def get_grid(self, step: int) -> xr.Dataset:
        """Get grid."""
        x = self.x[0::step]
        y = self.y[0::step]
        xs, ys = np.meshgrid(x, y)
        transformer = Transformer.from_crs(self.crs, WGS84)
        lats, lons = transformer.transform(xs, ys)
        return xr.Dataset(
            data_vars={
                "lon": (("y", "x"), lons),
                "lat": (("y", "x"), lats),
                "xs": (("y", "x"), xs),
                "ys": (("y", "x"), ys),
            },
            coords={
                "x": ("x", x),
                "y": ("y", y),
            },
            attrs={
                "crs": self.crs,
                "LL_x": xs[0, 0],
                "LL_y": ys[0, 0],
                "UR_x": xs[-1, -1],
                "UR_y": ys[-1, -1],
                "LR_x": xs[0, -1],
                "LR_y": ys[0, -1],
                "UL_x": xs[-1, 0],
                "UL_y": ys[-1, 0],
                "LL_lon": lons[0, 0],
                "LL_lat": lats[0, 0],
                "UR_lon": lons[-1, -1],
                "UR_lat": lats[-1, -1],
                "LR_lon": lons[0, -1],
                "LR_lat": lats[0, -1],
                "UL_lon": lons[-1, 0],
                "UL_lat": lats[-1, 0],
                "xsize": x.size,
                "ysize": y.size,
                "xscale": x[1] - x[0],
                "yscale": y[1] - y[0],
            },
        )


def reformat(
    t0: datetime,
    t1: datetime,
    basepath: Path,
    outpath: Path,
) -> None:
    """Reformat baltrad data files."""
    outpath = outpath / "radar"
    outpath.mkdir(parents=True, exist_ok=True)
    ti = t0
    while ti <= t1:
        reader = BaltradReader.from_datetime(ti, basepath)
        data = reader.get_data()
        data.attrs["crs"] = str(data.attrs["crs"])
        data.to_netcdf(
            outpath / ti.strftime("radar_%Y%m%d_%H_%M.nc"),
            encoding={var: {"zlib": True} for var in data.variables},
        )
        ti += timedelta(minutes=15)
