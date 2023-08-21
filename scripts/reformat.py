#!/usr/bin/env python
from datetime import datetime
from pathlib import Path
from sys import argv
import argparse

from pps_mw_training.pipelines.pr_nordic.data import baltrad


T0 = "2020-01-01T00:00:00"
T1 = "2020-01-31T00:00:00"


def cli(args_list: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Reformat Baltrad files."
    )
    parser.add_argument(
        "-b",
        "--basepath",
        dest="basepath",
        type=str,
        help="Base path to Baltrad dataset.",
        default="/tmp",
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        type=str,
        help=f"End date, default is {T1}",
        default=T1,
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
        "--start",
        dest="start",
        type=str,
        help=f"Start date, default is {T0}",
        default=T0,
    )
    args = parser.parse_args(args_list)
    t0 = datetime.fromisoformat(args.start)
    t1 = datetime.fromisoformat(args.end)
    basepath = Path(args.basepath)
    outpath = Path(args.outpath)
    baltrad.reformat(t0, t1, basepath, outpath)


if __name__ == "__main__":
    cli(argv[1:])
