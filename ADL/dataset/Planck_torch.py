import torch
import numpy as np
import pandas as pd
import os
from typing import List
from matplotlib import pyplot as plt


class Planck_Regression_Dataset(torch.utils.data.Dataset):
    """Planck_Regression_Dataset."""

    def __init__(self, data_path: str, target_path: str, pix2: List[int], patch_size: int = 32,
                 reg_prm: str = "M500"):
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

    def _prepare(self):
        """_prepare."""
        self.data = {}
        for i in self.pix2:
            self.data[i] = np.load(os.path.join(self.data_path, f"{i}.npy")).astype(np.float32)

        target = pd.read_csv(self.target_path)

        if not set(["x", "y", "pix2", self.reg_prm]).issubset(list(target)):
            raise(ValueError("Dataset table should have x, y, pix2 & reg_prm columns."))

        # Remove objects that are too close to grid and won't fit
        target = target[np.in1d(target["pix2"], self.pix2)]
        target = target[target["x"] >= self.patch_size]
        target = target[target["y"] >= self.patch_size]
        target = target[target["x"] < 2**11 - self.patch_size]
        target = target[target["y"] < 2**11 - self.patch_size]

        self.target = target

    def __getitem__(self, idx: int):
        """__getitem__.

        :param idx:
        :type idx: int
        """
        line = self.target.iloc[idx]
        x, y = line["x"], line["y"]
        size = self.patch_size // 2
        image = self.data[line["pix2"]]

        image = image[x - size: x + size, y - size: y + size]
        image = torch.tensor(image)
        image = torch.permute(image, [2, 0, 1])

        return image, line[self.reg_prm]

    def __len__(self):
        """__len__."""
        return len(self.target)

    def check_data(self, idx: int = 0):
        """check_data.

        :param idx:
        :type idx: int
        """
        X, y = self[idx]
        rows, cols = 2, 3
        f, ax = plt.subplots(rows, cols, figsize=(12, 8))
        for i in range(X.shape[-1]):
            ax[i // cols][i % cols].imshow(X[:, :, i])
        ax[0][0].set_title("{} = {:.2f}".format(self.reg_prm, y))

        f.tight_layout()
