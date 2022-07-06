"""Scripts for training models."""
from . import ADL_Unet
from ADL.dataset import Planck_Dataset
from ADL.model import pixels as p
import os


def train_Planck_Unet(model_name: str, data_path: str, target_path: str, model_path: str,
                      pixels: str, pretrained: str, batch_size: str, epochs: str,
                      device: str, continue_train: bool = False, lr_scheduler: str = None) -> None:
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

    model = ADL_Unet(os.path.join(model_path, model_name + "_ep{epoch:03}.hdf5"), weights=weights,
                     lr_scheduler=lr_scheduler)
    model.train(dataset_train, dataset_val, int(epochs), continue_train=continue_train)
