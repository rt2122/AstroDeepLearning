import argparse
import os
import time
import datetime
from .scripts import preproc_HFI_Planck, generate_masks_and_patches_Planck

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script for HFI Planck data.")
    parser.add_argument("command", metavar="<command>",
                        help="'preproc' for preparing HFI Planck data.\n"
                             "'target' for generating masks and patches for Planck.")
    parser.add_argument("--inpath", metavar="<inpath>", default=".",
                        help="Input path.")
    parser.add_argument("--outpath", metavar="<outpath>", default=".", help="Output path.")
    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)

    start_time = time.time()
    if args.command == "preproc":
        preproc_HFI_Planck(args.inpath, args.outpath)
    elif args.command == "target":
        generate_masks_and_patches_Planck(args.inpath, args.outpath)
    else:
        print("Command is not recognized.")

    finish_time = time.time()
    diff = str(datetime.timedelta(seconds=finish_time - start_time))
    print(f"Planck {args.command} completed in {diff}")
