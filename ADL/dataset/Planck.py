"""Module for Planck Dataset."""
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Union, Iterator
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
import warnings
from .dataset import do_aug


def split_dataframe(df: pd.DataFrame, batch_size: int) -> List[pd.DataFrame]:
    """Split DataFrame into batches of equal size.

    :param df: DataFrame to split.
    :type df: pd.DataFrame
    :param batch_size: Size of batches.
    :type batch_size: int
    :rtype: List[pd.DataFrame]
    """
    batches = []
    n_batches = len(df) // batch_size + 1
    for i in range(n_batches):
        batches.append(df[i * batch_size : (i + 1) * batch_size])
    return batches


class Planck_Dataset:
    """Dataset for Planck HFI data.

    Organization for directories:
    ::

        data_path
        ├── 0.npy
        ├── 1.npy
        ...
        └── 47.npy
        target_path
        ├── 0.npy
        ├── 1.npy
        ...
        ├── 47.npy
        ├── pc.csv
        ├── descr.txt
        └── cats
            ├── PSZ2_z.csv
            ...
            └── MCXCwp.csv


    | i.npy - data corresponding to i'th tile of HEALPix partition with nside=2 and
      nested scheme.
    | pc.csv - table with coordinates of patches (top left corner).
    | descr.txt - description of dataset (optional).

    | pc.csv:

    +------------+------------+------------------+
    | x          | y          | pix2             |
    +============+============+==================+
    | (X coord)  | (Y coord)  | (Index of pixel) |
    +------------+------------+------------------+
    | ...        | ...        | ...              |
    +------------+------------+------------------+

    | cats - directory with catalogs in .csv format. Each catalog has columns: [RA, DEC, z, M500].

    :param data_path: Path for Planck HFI data divided into 48 tiles.
    :type data_path: str
    :param target_path: Path for masks and patch coordinates.
    :type target_path: str
    :param pix2: List of pixels of nside=2 HEALPix partition.
    :type pix2: List[int]
    :param batch_size: Size of batch.
    :type batch_size: int
    :param patch_size: Size of path.
    :type patch_size: int
    :param shuffle: Flag for shuffling dataset.
    :type shuffle: bool
    :param augmentation: Augmenter. Use "default" preset for default augmentation
        (horizontal and vertical flips + rotation by an angle multiple of 90 degrees).
    :type augmentation: Union[iaa.Augmenter, str]
        :param lfi_path: Path to LFI data (the same way as HFI data).
        :type lfi_path: str
    """

    def __init__(
        self,
        data_path: str,
        target_path: str,
        pix2: List[int],
        batch_size: int,
        patch_size: int = 64,
        shuffle: bool = False,
        augmentation: Union[iaa.Augmenter, str] = "default",
        lfi_path: str = None,
    ):
        """Initialize dataset."""
        self.data_path = data_path
        self.target_path = target_path
        self.pix2 = pix2
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.lfi_path = lfi_path
        if type(augmentation) == str:
            if augmentation == "default":
                self.augmentation = iaa.SomeOf(
                    (0, 4),
                    [
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.OneOf(
                            [
                                iaa.Affine(rotate=90),
                                iaa.Affine(rotate=180),
                                iaa.Affine(rotate=270),
                            ]
                        ),
                    ],
                )
            else:
                warnings.warn(
                    "Wrong preset name for augmentation."
                    " Continuing without augmentation."
                )
                self.augmentation = None
        self._prepare()

    def _prepare(self) -> None:
        """Load data.

        :rtype: None
        """
        self.data = {}
        self.target = {}
        for i in self.pix2:
            self.data[i] = np.load(os.path.join(self.data_path, f"{i}.npy"))
            self.target[i] = np.load(os.path.join(self.target_path, f"{i}.npy"))

            if self.lfi_path:
                self.data[i] = np.dstack(
                    [np.load(os.path.join(self.lfi_path, f"{i}.npy")), self.data[i]]
                )
        coords = pd.read_csv(os.path.join(self.target_path, "pc.csv"))
        coords = coords[np.in1d(coords["pix2"], self.pix2)]
        coords.index = np.arange(len(coords))
        self.coords = coords
        self._split_batches()

    def _split_batches(self) -> None:
        """Split patches coords into batches.

        :rtype: None
        """
        coords = self.coords
        if self.shuffle:
            coords = coords.sample(frac=1)
        self.batches = split_dataframe(coords, self.batch_size)
        if len(self.batches[-1]) == 0:
            self.batches = self.batches[:-1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        """Get batch.

        :param idx:
        :type idx: int
        :rtype: Tuple[np.ndarray]
        """
        batch = self.batches[idx]
        hsize = self.patch_size / 2

        X = []
        Y = []
        for x, y, pix in zip(batch["x"], batch["y"], batch["pix2"]):
            X.append(self.data[pix][x - hsize : x + hsize, y - hsize : y + hsize, :])
            Y.append(self.target[pix][x - hsize : x + hsize, y - hsize : y + hsize, :])

        X = np.array(X)
        Y = np.array(Y)

        if self.augmentation:
            X, Y = do_aug(X, Y, self.augmentation)

        return X, Y

    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)

    def generator(self) -> Iterator[Tuple[np.ndarray]]:
        """Generate batches for training.

        :rtype: Iterator[Tuple[np.ndarray]]
        """
        for i in range(len(self)):
            yield self[i]
        if self.shuffle:
            self._split_batches()

    def check_data(
        self,
        idx: int = 0,
        batch_idx: int = 0,
        X: np.ndarray = None,
        Y: np.ndarray = None,
        pred: np.ndarray = None,
        name: str = "",
    ) -> None:
        """Fast check of data in dataset.

        :param idx: Index of batch.
        :type idx: int
        :param batch_idx: Index within batch.
        :type batch_idx: int
        :param X: Batch of images.
        :type X: np.ndarray
        :param Y: Batch of masks.
        :type Y: np.ndarray
        :rtype: None
        """
        if X is None or Y is None:
            X, Y = self[idx]
        X = X[batch_idx]
        Y = Y[batch_idx]
        if X.shape[-1] == 6:
            rows, cols = 3, 3
        elif X.shape[-1] == 9:
            rows, cols = 4, 3
        else:
            raise (ValueError("X shape incorrect"))
        f, ax = plt.subplots(rows, cols, figsize=(10, 10))
        for i in range(X.shape[-1]):
            ax[i // cols][i % cols].imshow(X[:, :, i])
        if Y.shape[-1] == 1:
            ax[rows - 1][0].imshow(Y)
        else:
            ax[rows - 1][0].imshow(255 * Y[:, :, [0, 0, 1]])
        ax[rows - 1][0].set_xlabel("Ground truth")

        if pred is not None:
            ax[rows - 1][1].imshow(pred[batch_idx])
            ax[rows - 1][1].set_xlabel(
                "Prediction max {:.2f}".format(pred[batch_idx].max())
            )
            ax[rows - 1][2].imshow([[0]])
            ax[rows - 1][2].set_xlabel(name)

        f.tight_layout()
