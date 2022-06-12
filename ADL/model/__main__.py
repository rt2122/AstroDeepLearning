"""Module for training models."""
import argparse
import time
import datetime
from .scripts import train_Planck_Unet


def make_parser() -> argparse.ArgumentParser:
    """Create parser.

    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Training script for Unet model & Planck data.")
    parser.add_argument("command", metavar="<command>",
                        help="'train' for training model.")
    parser.add_argument("model_name", metavar="<model_name>",
                        help="Code name for the model.")
    parser.add_argument("data_path", metavar="<data_path>",
                        help="Directory with 48 tiles of HFI Planck data.")
    parser.add_argument("target_path", metavar="<target_path>",
                        help="Directory with masks for Planck.")
    parser.add_argument("model_path", metavar="<model_path>",
                        help="Directory for saving models.")
    parser.add_argument("--pixels", metavar="<pixels>", default="default",
                        help="Distribution of HEALPix pixels between train, val & test.")
    parser.add_argument("--pretrained", metavar="<pretrained>", default="",
                        help="Path to pretrained weights.")
    parser.add_argument("--epochs", metavar="<epochs>", default="100",
                        help="Number of epochs.")
    parser.add_argument("--batch_size", metavar="<batch_size>", default="20",
                        help="Size of batch.")
    parser.add_argument("--device", metavar="<device>", default="cpu",
                        help="Device for training. 'cpu' or 'gpu'.")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    start_time = time.time()

    if args.command == "train":
        train_Planck_Unet(args.model_name, args.data_path, args.target_path, args.model_path,
                          args.pixels, args.pretrained, args.batch_size, args.epochs, args.device)
    else:
        print("Command is not recognized.")

    finish_time = time.time()
    diff = str(datetime.timedelta(seconds=finish_time - start_time))
    print(f"Planck {args.command} completed in {diff}")
