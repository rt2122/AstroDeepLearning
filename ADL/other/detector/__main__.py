"""Module for detecting objects in Planck masks."""
import argparse
from argparse import RawTextHelpFormatter
import time
import datetime
from .scripts import scan_Planck_Unet, extract_cat_Planck


def make_parser() -> argparse.ArgumentParser:
    """Create parser.

    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Script for detector on Planck data.",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("command", metavar="<command>",
                        help="'scan' for scanning Planck data with selected model.\n"
                        "'cat' for creating catalog from scan."
                        )
    parser.add_argument("in_path", metavar="<in_path>",
                        help="Path to hdf5 file to Unet Planck model (for 'scan').\n"
                        "Path to directory with scans (for 'cat')."
                        )
    parser.add_argument("out_path", metavar="<out_path>",
                        help="Output directory for scans.")
    parser.add_argument("--data_path", metavar="<data_path>", default='.',
                        help="Path to Planck data in HEALPix tiles (for 'scan').")
    parser.add_argument("--step", metavar="<step>", default="64",
                        help="Size of step for scanning. Should be 2^n and <= 64.\n"
                        "Or 'fast' for scanning each tile at once (detects less objects)."
                        )
    parser.add_argument("--device", metavar="<device>", default="cpu",
                        help="Device for scanning. 'cpu' or 'gpu'.")
    parser.add_argument("--thr", metavar="<thr>", default="0.1",
                        help="Threshold for 'cat'.")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    start_time = time.time()

    if args.command == "scan":
        scan_Planck_Unet(args.in_path, args.data_path, args.out_path, args.step, args.device)
    elif args.command == "cat":
        extract_cat_Planck(args.in_path, args.out_path, args.thr)
    else:
        print("Command is not recognized.")

    finish_time = time.time()
    diff = str(datetime.timedelta(seconds=finish_time - start_time))
    print(f"Planck {args.command} completed in {diff}")
