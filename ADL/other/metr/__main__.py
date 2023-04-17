"""Module for calculating metrics for catalogs."""
import argparse
from argparse import RawTextHelpFormatter
import time
import datetime
from .scripts import calc_prec_recall_by_range_parameter


def make_parser() -> argparse.ArgumentParser:
    """Create parser.

    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Script for calculating metrics of catalogs.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "det_path", metavar="<det_path>", help="Path to detected catalog."
    )
    parser.add_argument(
        "true_dir_path",
        metavar="<true_dir_path>",
        help="Path to directory with ground truth catalogs.",
    )
    parser.add_argument(
        "out_path", metavar="<out_path>", help="Path to output file with metrics."
    )
    parser.add_argument(
        "--rules",
        metavar="<rules>",
        default="",
        help="Rules preset for filtering catalogs.\n"
        "'east' or 'west' for east or west half of the sky.\n"
        "'b20' for cutting galactic area.\n"
        "example: 'east_b20' for east part of extragalactic sky.",
    )
    parser.add_argument(
        "--range_prm",
        metavar="<range_prm>",
        default="max_pred",
        help="Name of parameter for ranging depending on detected catalog.\n"
        "'max_pred', 's/n' or something else.",
    )
    parser.add_argument(
        "--range_preset",
        metavar="<range_preset>",
        default="linear",
        help="Values for range prm. 'linear' or 'brcat' or 'quantile'",
    )
    parser.add_argument(
        "--pixels",
        metavar="<pixels>",
        default="all",
        help="HEALPix pixels with nside=2.",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    start_time = time.time()

    calc_prec_recall_by_range_parameter(
        args.det_path,
        args.true_dir_path,
        args.out_path,
        args.rules,
        args.range_prm,
        args.range_preset,
        args.pixels,
    )

    finish_time = time.time()
    diff = str(datetime.timedelta(seconds=finish_time - start_time))
    print(f"Metrics calculated in {diff}")
