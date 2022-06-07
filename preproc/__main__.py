import argparse
import os
import time
import datetime
from .scripts import preproc_HFI_Planck

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing script for astronomical data.")
    parser.add_argument("data_name", metavar="<data_name>",
                        help="'Planck' (other data will be added later)")
    parser.add_argument("command", metavar="<command>",
                        help="'preproc' for preparing HFI Planck data.\n"
                             "'target' for generating masks and patches for Planck.")
    parser.add_argument("--inpath", metavar="<inpath>", default=".",
                        help="Input path.")
    parser.add_argument("--outpath", metavar="<outpath>", default=".", help="Output path.")

    args = parser.parse_args()
    start_time = time.time()
    if args.data_name == "Planck":
        if args.command == "preproc":
            if not os.path.exists(args.outpath):
                os.mkdir(args.outpath)
            preproc_HFI_Planck(args.inpath, args.outpath)
            finish_time = time.time()
            diff = str(datetime.timedelta(seconds=finish_time - start_time))
            print(f"Planck preprocessing completed in {diff}")
        elif args.command == "target":
            print("Planck target")
        else:
            print("Command is not recognized.")
    else:
        print("Data name is not recognized.")
