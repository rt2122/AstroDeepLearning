"""Module for pytorch models."""
import torch
from torch.utils.data import Sampler
import numpy as np
import pandas as pd
import os
from typing import List, Iterator
from matplotlib import pyplot as plt
import matplotlib
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
import random
import torchvision.transforms.functional as TF


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles: List[float]):
        """Initialize.

        :param angles: Angles.
        :type angles: List[float]
        """
        self.angles = angles

    def __call__(self, x):
        """Perform the transformation.

        :param x: Tensor on which to apply the transformation.
        """
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Planck_Regression_Dataset(torch.utils.data.Dataset):
    """Planck_Regression_Dataset."""

    def __init__(
        self,
        data_path: str,
        target_path: str,
        pix2: List[int],
        patch_size: int = 32,
        reg_prm: str = "M500",
    ):
        """Initialize dataset.

        :param data_path: Planck data path (npy files).
        :type data_path: str
        :param target_path: Target path (csv file which has x, y, pix2, reg_prm columns).
        :type target_path: str
        :param pix2: List of big pixels for this dataset.
        :type pix2: List[int]
        :param patch_size: Size of patch.
        :type patch_size: int
        :param reg_prm: Name for regression parameter.
        :type reg_prm: str
        """
        self.data_path = data_path
        self.target_path = target_path
        self.reg_prm = reg_prm
        self.pix2 = pix2
        self.patch_size = patch_size
        self._prepare()
        self.transforms = Compose(
            [
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                MyRotationTransform([90 * i for i in range(4)]),
            ]
        )

    def _prepare(self):
        """Load data."""
        self.data = {}
        for i in self.pix2:
            self.data[i] = np.load(os.path.join(self.data_path, f"{i}.npy")).astype(
                np.float32
            )

        target = pd.read_csv(self.target_path)

        if not set(["x", "y", "pix2", self.reg_prm]).issubset(list(target)):
            raise (
                ValueError("Dataset table should have x, y, pix2 & reg_prm columns.")
            )

        # Remove objects that are too close to grid and won't fit
        target = target[np.in1d(target["pix2"], self.pix2)]
        hsize = self.patch_size // 2
        target = target[target["x"] >= hsize]
        target = target[target["y"] >= hsize]
        max_size = self.data[self.pix2[0]].shape[0]
        target = target[target["x"] < max_size - hsize]
        target = target[target["y"] < max_size - hsize]
        target.index = np.arange(len(target))

        self.target = target

    def __getitem__(self, idx: int):
        """Get data by index.

        :param idx: Index.
        :type idx: int
        """
        line = self.target.iloc[idx]
        x, y = line["x"], line["y"]
        size = self.patch_size // 2
        image = self.data[line["pix2"]]

        image = image[x - size : x + size, y - size : y + size]
        image = torch.tensor(image)
        image = torch.permute(image, [2, 0, 1])
        image = self.transforms(image)

        return image, line[self.reg_prm]

    def __len__(self):
        """Get size of dataset."""
        return len(self.target)

    def check_data(self, idx: int = 0):
        """Check data with matplotlib.

        :param idx:
        :type idx: int
        """
        X, y = self[idx]
        rows, cols = 2, 3
        f, ax = plt.subplots(rows, cols, figsize=(12, 8))
        for i in range(X.shape[0]):
            ax[i // cols][i % cols].imshow(X[i, :, :])
        ax[0][0].set_title("{} = {:.2f}".format(self.reg_prm, y))

        f.tight_layout()

    def target_prm_histogram(self, ax: matplotlib.axes, n_bins: int = 20):
        """Show histogram for target prm.

        :param ax: Where to show.
        :type ax: matplotlib.axes
        :param n_bins: Number of bins.
        :type n_bins: int
        """
        ax.hist(self.target[self.reg_prm], bins=n_bins)
        ax.set_xlabel(self.reg_prm)
        ax.grid()


class StratifiedSampler(Sampler[int]):
    """StratifiedSampler."""

    def __init__(
        self,
        data_source: Planck_Regression_Dataset,
        n_bins: int = 5,
        batch_size: int = 128,
        n_batches: int = 30,
    ):
        """Initialize.

        :param data_source:
        :type data_source: Planck_Regression_Dataset
        :param n_bins:
        :type n_bins: int
        :param batch_size:
        :type batch_size: int
        :param n_batches:
        :type n_batches: int
        """
        self.data_source = data_sourceADL / dataset / Planck_torch.py
        self.n_bins = n_bins
        target_df = data_source.target
        target_df["index"] = target_df.index
        prm = data_source.reg_prm
        target_vals, bins = pd.cut(
            target_df[prm],
            target_df[prm].quantile(np.linspace(0, 1, n_bins + 1)),
            retbins=True,
        )
        self.bins = bins
        self.grouped_df = target_df.groupby(target_vals)
        self.group_size = len(target_df)
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self) -> Iterator[int]:
        """Iterate.

        :rtype: Iterator[int]
        """
        sample = []
        for i in range(self.n_batches):
            sample.append(
                self.grouped_df.apply(
                    lambda x: x.sample(n=self.batch_size // self.n_bins, replace=True)
                )
            )
        sample = pd.concat(sample)
        return iter(sample["index"])

    def __len__(self):
        """Size of dataset."""
        return self.group_size * self.n_bins
