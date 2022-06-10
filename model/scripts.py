from . import ADL_Unet
from AstroDeepLearning.dataset import Planck_Dataset
from AstroDeepLearning.model import pixels as p

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_Planck_Unet(data_path: str, target_path: str, model_path: str, pixels: str,
                      pretrained: str, batch_size: str, epochs: str) -> None:
    """Full process of training.

    :param data_path: Path to data.
    :type data_path: str
    :param target_path: Path to target.
    :type target_path: str
    :param model_path: Path to model.
    :type model_path: str
    :param pixels: Pixels preset.
    :type pixels: str
    :param pretrained: Path to pretrained models.
    :type pretrained: str
    :param batch_size: Size of batch.
    :type batch_size: str
    :param epochs: Number of epochs
    :type epochs: str
    :rtype: None
    """

    if pixels in dir(p) and type(getattr(p, pixels)) == dict:
        pix_dict = getattr(p, pixels)
    else:
        print("Pixels parameter is not recognized.")
        return

    weights = None
    if pretrained != "":
        weights = pretrained
    dataset_train = Planck_Dataset(data_path=data_path, target_path=target_path,
                                   pix2=pix_dict["train"], batch_size=int(batch_size))
    dataset_train.prepare()
    dataset_val = Planck_Dataset(data_path=data_path, target_path=target_path,
                                 pix2=pix_dict["val"], batch_size=int(batch_size))
    dataset_val.prepare()

    model = ADL_Unet(model_path, weights=weights)
    model.train(dataset_train, dataset_val, int(epochs))
