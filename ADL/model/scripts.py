"""Scripts for training models."""
from . import ADL_Unet
from ADL.dataset import Planck_Dataset
from ADL.model import pixels as p
import os


def train_Planck_Unet(model_name: str, data_path: str, target_path: str, model_path: str,
                      pixels: str, pretrained: str, batch_size: str, epochs: str,
                      device: str, continue_train: bool = False, lr_scheduler: str = None,
                      save_best_only: bool = False, test_as_val: bool = False,
                      n_filters: int = 8, n_blocks: int = 5) -> None:
    """Full process of training.

    :param data_path: Path to data.
    :type data_path: str
    :param target_path: Path to target.
    :type target_path: str
    :param model_path: Path to models' dir.
    :type model_path: str
    :param pixels: Pixels preset.
    :type pixels: str
    :param pretrained: Path to pretrained models.
    :type pretrained: str
    :param batch_size: Size of batch.
    :type batch_size: str
    :param epochs: Number of epochs
    :type epochs: str
    :param device: Device for training. 'cpu' or 'gpu'.
    :type device: str
    :param continue_train: Find history file and get number of last epoch.
    :type continue_train: bool
    :param lr_scheduler: LR scheduler.
    :type lr_scheduler: str
    :param save_best_only: Flag for saving only best models.
    :type save_best_only: bool
    :param test_as_val: Flag for adding test metrics.
    :type test_as_val: bool
    :param n_filters: Number of filters in block.
    :type n_filters: int
    :param n_blocks: Number of blocks.
    :type n_blocks: int
    :rtype: None
    """
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if pixels in dir(p) and type(getattr(p, pixels)) == dict:
        pix_dict = getattr(p, pixels)
    else:
        print("Pixels parameter is not recognized.")
        return

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    weights = None
    if pretrained != "":
        weights = pretrained
    dataset_train = Planck_Dataset(data_path=data_path, target_path=target_path,
                                   pix2=pix_dict["train"], batch_size=int(batch_size), shuffle=True)
    dataset_train.prepare()
    dataset_val = Planck_Dataset(data_path=data_path, target_path=target_path,
                                 pix2=pix_dict["val"], batch_size=int(batch_size))
    dataset_val.prepare()

    dataset_test = None
    if test_as_val:
        dataset_test = Planck_Dataset(data_path=data_path, target_path=target_path,
                                      pix2=pix_dict["test"], batch_size=int(batch_size))
        dataset_test.prepare()

    model = ADL_Unet(os.path.join(model_path, model_name + "_ep{epoch:03}.hdf5"), weights=weights,
                     lr_scheduler=lr_scheduler, save_best_only=save_best_only,
                     test_as_val=dataset_test, n_filters=n_filters, n_blocks=n_blocks)
    model.train(dataset_train, dataset_val, int(epochs), continue_train=continue_train)
