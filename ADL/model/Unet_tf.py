"""Module for Unet model (tensorflow) and Planck data."""
import numpy as np
import pandas as pd
import os

from ADL.dataset import Planck_Dataset

from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D,
                                     Activation, BatchNormalization)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.losses import binary_crossentropy

from typing import Tuple


def iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Intersection over Union.

    :param y_pred: Predictioin mask.
    :type y_pred: np.ndarray
    :param y_true: Ground truth mask.
    :type y_true: np.ndarray
    :rtype: float
    """
    iou_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        iou_sum += inters / union
    return iou_sum


def dice(y_pred: np.ndarray, y_true: np.ndarray, eps=0.1) -> float:
    """Dice coefficient.

    :param y_pred: Predictioin mask.
    :type y_pred: np.ndarray
    :param y_true: Ground truth mask.
    :type y_true: np.ndarray
    :param eps: Epsilon.
    :rtype: float
    """
    dice_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        dice_sum += K.mean((2 * inters + eps) / (union + eps))
    return dice_sum


class ADL_Unet:
    """Unet model.

    :param model_path: Template path for saving weights of model. Metrics & loss variables are
        available. Example: /path/to/models/Unet-val_loss{val_loss:.3f}-ep{epoch}.hdf5
    :type model_path: str
    :param input_shape: Shape of input data. Default for Planck data.
    :type input_shape: Tuple[int]
    :param n_filters: Number of filters in block.
    :type n_filters: int
    :param n_blocks: Number of blocks.
    :type n_blocks: int
    :param n_output_layers: Number of output layers.
    :type n_output_layers: int
    :param lr: Learning rate.
    :type lr: float
    :param add_batch_norm: Flag for batch normalization.
    :type add_batch_norm: bool
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param weights: Path to pretrained weights.
    :type weights: str
    """

    def __init__(self, model_path: str, input_shape: Tuple[int] = (64, 64, 6), n_filters: int = 8,
                 n_blocks: int = 5, n_output_layers: int = 1, lr: float = 1e-4,
                 add_batch_norm: bool = False, dropout_rate: float = 0.2, weights: str = None):
        """Initialize."""
        self.model = Unet_model(input_shape, n_filters, n_blocks, n_output_layers, lr,
                                add_batch_norm, dropout_rate, weights)
        self.checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                          save_best_only=False, mode='min', save_weights_only=False)

        self.model_path = model_path
        self.history = []

    def train(self, trainset: Planck_Dataset, valset: Planck_Dataset, n_epochs: int,
              init_epoch: int = 0) -> None:
        """Train model.

        :param trainset: Dataset for training.
        :type trainset: Planck_Dataset
        :param valset: Dataset for validation.
        :type valset: Planck_Dataset
        :param n_epochs: Number of epochs.
        :type n_epochs: int
        :param init_epoch: Index of initial epoch.
        :type init_epoch: int
        :rtype: None
        """
        for i in range(init_epoch, init_epoch + n_epochs):
            print(f"Epoch #{i}")
            history = self.model.fit(trainset.generator(), epochs=i+1, verbose=1,
                                     callbacks=[self.checkpoint],
                                     validation_data=valset.generator(), initial_epoch=i)
            self.history.append(history.history)
            self.save_history()

    def save_history(self) -> None:
        """Save history file.

        :rtype: None
        """
        df = pd.concat(map(lambda x: pd.DataFrame(x, index=[0]), self.history))
        df.index = np.arange(1, len(df) + 1)
        df.index.name = "epoch"
        df.to_csv(os.path.join(os.path.dirname(self.model_path), 'history.csv'))

    def make_prediction(self, dataset: Planck_Dataset, idx: int = 0) -> Tuple[np.ndarray]:
        """Make prediction for one batch.

        :param dataset: Dataset for prediction.
        :type dataset: Planck_Dataset
        :param idx: Index of batch.
        :type idx: int
        :rtype: Tuple[np.ndarray]
        """
        X, Y = dataset[idx]
        pred = self.model.predict(X)
        return X, Y, pred


def Unet_model(input_shape: Tuple[int] = (64, 64, 6), n_filters: int = 8, n_blocks: int = 5,
               n_output_layers: int = 1, lr: float = 1e-4, add_batch_norm: bool = False,
               dropout_prm: float = 0.2, weights: str = None) -> Model:
    """Create tensorflow model Unet.

    :param input_shape: Shape of input data. Default for Planck data.
    :type input_shape: Tuple[int]
    :param n_filters: Number of filters in block.
    :type n_filters: int
    :param n_blocks: Number of blocks.
    :type n_blocks: int
    :param n_output_layers: Number of output layers.
    :type n_output_layers: int
    :param lr: Learning rate.
    :type lr: float
    :param add_batch_norm: Flag for batch normalization.
    :type add_batch_norm: bool
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param weights: Path to pretrained weights.
    :type weights: str
    :rtype: Model
    """
    if weights is not None:
        model = load_model(weights, custom_objects={'iou': iou, 'dice': dice})
        model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy',
                                                                                iou, dice])
        return model

    encoder = []
    inputs = Input(input_shape)
    prev = inputs
    for i in range(n_blocks):
        cur = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same',
                     kernel_initializer='he_normal')(prev)
        if add_batch_norm:
            cur = BatchNormalization()(cur)
        else:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)

        cur = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same',
                     kernel_initializer='he_normal')(cur)

        if add_batch_norm:
            cur = BatchNormalization()(cur)
        else:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)

        encoder.append(cur)

        cur = MaxPooling2D(padding='valid')(cur)

        n_filters *= 2
        prev = cur

    for i in range(n_blocks - 1, -1, -1):
        cur = UpSampling2D()(prev)
        cur = Conv2D(filters=n_filters, kernel_size=3, padding='same')(cur)
        if not add_batch_norm:
            cur = Dropout(dropout_prm)(cur)
        cur = Activation(relu)(cur)
        cur = concatenate([cur, encoder[i]], axis=3)

        cur = Conv2D(filters=n_filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)
        if not add_batch_norm:
            cur = Dropout(dropout_prm)(cur)

        prev = cur
        n_filters //= 2

    prev = Conv2D(n_output_layers, kernel_size=3, padding='same')(prev)
    prev = Activation(sigmoid)(prev)

    model = Model(inputs=inputs, outputs=prev)
    model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy', iou, dice])
    return model
