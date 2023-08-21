#!/usr/bin/env python
from enum import Enum
from pathlib import Path
from sys import argv
from typing import Sequence
import argparse
import logging


from pps_mw_training.pipelines.pr_nordic.data import utils
from pps_mw_training.pipelines.pr_nordic.data.atms import AtmsL1bReader
from pps_mw_training.pipelines.pr_nordic.data.baltrad import BaltradReader
from pps_mw_training.pipelines.pr_nordic.data.regridder import Regridder
from pps_mw_training.pipelines.pr_nordic.settings import INPUT_PARAMS


CHUNKSIZE = 16


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def regrid_files(
    level1b_files: list[Path],
    baltrad_file: Path,
    grid_step: int,
    method: str,
    chunk_size,
    channels: Sequence[Enum],
    outpath: Path,
) -> None:
    """Regrid files."""
    baltrad_reader = BaltradReader(baltrad_file)
    grid = baltrad_reader.get_grid(grid_step)
    for files in utils.reshape_filelist(level1b_files, chunk_size):
        logging.info(f"Start processing {files[0]}.")
        data = AtmsL1bReader.get_data(files)
        data = Regridder(data, grid).regrid(channels, method=method)
        if data is not None:
            outfile = utils.Writer(data, outpath).write()
            logging.info(f"Wrote {outfile} to disc.")
        else:
            logging.warning("No outfile was written.")
        logging.info(f"Done processing  {files[0]}.")


def cli(args_list: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Regrid data from ATMS onto the Baltrad grid."
    )
    parser.add_argument(
        dest="level1b_files",
        type=str,
        nargs='+',
        help="Full path to ATMS level1b file(s)",
    )
    parser.add_argument(
        dest="baltrad_file",
        type=str,
        help="Full path to a Baltrad pn150 file.",
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        type=str,
        choices=["linear", "nearest"],
        help="Interpolation method",
        default="linear",

    )
    parser.add_argument(
        "-o",
        "--outpath",
        dest="outpath",
        type=str,
        help="Path where to write data from processed files.",
        default="/tmp",
    )
    parser.add_argument(
        "-s",
        "--grid-step",
        dest="grid_step",
        type=int,
        help="Grid step for regridding, e.g. 4 means every forth position",
        default=4,
    )
    args = parser.parse_args(args_list)
    regrid_files(
        level1b_files=[Path(f) for f in args.level1b_files],
        baltrad_file=Path(args.baltrad_file),
        grid_step=args.grid_step,
        method=args.method,
        chunk_size=CHUNKSIZE,
        channels=utils.get_channels(INPUT_PARAMS),
        outpath=Path(args.outpath),
    )


if __name__ == "__main__":
    cli(argv[1:])
