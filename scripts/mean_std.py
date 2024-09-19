import xarray as xr
import glob
from pathlib import Path
import os
from typing import Any, Optional
import numpy as np
import pickle
from pps_mw_training.pipelines.cloud_base.settings import INPUT_PARAMS
from tqdm import tqdm


def update_std_mean(
    input_files: list[Path], input_params: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """calculate std and mean for the input dataset to normalise"""

    all_dict = {}
    with xr.open_dataset(input_files[0]) as ds:
        for key in ds.keys():
            all_dict[key] = {"n": 0, "x": 0, "x2": 0}

    for file in tqdm(input_files):
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

def split_files(files):
    for ix, file in enumerate(files):
        print(ix)
        with xr.open_dataset(file) as ds:
            for i in range(len(ds.nscene)):
                outfile = os.path.basename(file)[:-3] + f"_{i}.nc"
                if np.sum(ds.sel(nscene=i).cloud_base.values < 60):
                    ds.sel(nscene=i).to_netcdf(os.path.join(outpath, outfile))


files = glob.glob("/nobackup/smhid20/users/sm_indka/collocated_data/split_data/*cnn*.nc")
files1 = glob.glob("/nobackup/smhid20/users/sm_indka/collocated_data/split_data/filtered_data/*cnn*.nc")
outpath = "/nobackup/smhid20/users/sm_indka/collocated_data/split_data/"

print(len(files))

files = files + files1
input_params = update_std_mean(files, INPUT_PARAMS)

with open("input_params.pickle", 'wb') as f:
    pickle.dump(input_params, f)
