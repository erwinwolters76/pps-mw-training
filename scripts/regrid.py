#!/usr/bin/env python
from pathlib import Path
from sys import argv
from typing import List
import argparse
import logging

from pps_mw_training.pipelines.pr_nordic.data.atms import AtmsL1bReader
from pps_mw_training.pipelines.pr_nordic.data.baltrad import BaltradReader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def regrid_files(
    level1b_files: list[Path],
    baltrad_file: Path,
    grid_step: int,
    outpath: Path,
) -> None:
    """Regrid files."""
    baltrad_reader = BaltradReader(baltrad_file)
    grid = baltrad_reader.get_grid(grid_step)
    print(grid)
    for level1b_file in level1b_files:
        logging.info(f"Start processing {level1b_file}.")
        atms_reader = AtmsL1bReader(level1b_file)
        data = atms_reader.get_data()
        print(data)
        # {TODO: add regeridding and writing of data}
        logging.info(f"Done processing  {level1b_file}.")


def cli(args_list: List[str]) -> None:
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
        "-s",
        "--grid-step",
        dest="grid_step",
        type=int,
        help="Grid step for regridding, e.g. 4 means every forth position",
        default=4,
    )
    parser.add_argument(
        "-o",
        "--outpath",
        dest="outpath",
        type=str,
        help="Path where to write data from processed files.",
        default="/tmp",
    )
    args = parser.parse_args(args_list)
    level1b_files = [Path(f) for f in args.level1b_files]
    baltrad_file = Path(args.baltrad_file)
    outpath = Path(args.outpath)
    regrid_files(level1b_files, baltrad_file, args.grid_step, outpath)


if __name__ == "__main__":
    cli(argv[1:])
