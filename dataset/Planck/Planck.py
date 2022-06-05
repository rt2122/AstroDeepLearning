import numpy as np
import pandas as pd
import os
from typing import List, Tuple


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
        batches.append(df[i*batch_size: (i + 1) * batch_size])
    return batches


class Planck_Dataset:
    """Dataset for Planck HFI data.

    Organization for directories:
    ::

        datapath
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
        └── descr.txt

    :param datapath: Path for Planck HFI data divided into 48 tiles.
    :type datapath: str
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
    """

    def __init__(self, datapath: str, target_path: str, pix2: List[int], batch_size: int,
                 patch_size: int = 64, shuffle: bool = False):
        """Initialize dataset.
        """
        self.datapath = datapath
        self.target_path = target_path
        self.pix2 = pix2
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.shuffle = shuffle

    def prepare(self) -> None:
        """Load data.

        :rtype: None
        """
        self.data = {}
        self.target = {}
        for i in self.pix2:
            self.data[i] = np.load(os.path.join(self.datapath, f"{i}.npy"))
            self.target[i] = np.load(os.path.join(self.target_path, f"{i}.npy"))
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
            coords = coords.sample(1)
        self.batches = split_dataframe(coords, self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        """__getitem__.

        :param idx:
        :type idx: int
        :rtype: Tuple[np.ndarray]
        """
        batch = self.batches[idx]
        size = self.patch_size

        X = []
        Y = []
        for x, y, pix in zip(batch["x"], batch["y"], batch["pix2"]):
            X.append(self.data[pix][x:x+size, y:y+size, :])
            Y.append(self.target[pix][x:x+size, y:y+size, :])
        return np.array(X), np.array(Y)
