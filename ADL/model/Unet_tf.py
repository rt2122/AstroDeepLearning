"""Module for Unet model (tensorflow) and Planck data."""
import numpy as np
import pandas as pd
import os

from ADL.dataset import Planck_Dataset

from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D,
                                     BatchNormalization, Layer, Conv2DTranspose)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

from typing import Tuple, Dict, Union, List


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
    :param n_classes: Number of output layers.
    :type n_classes: int
    :param lr: Learning rate.LearningRateScheduler
    :type lr: float
    :param add_batch_norm: Flag for batch normalization.
    :type add_batch_norm: bool
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param weights: Path to pretrained weights.
    :type weights: str
    :param lr_scheduler: LR preset or dictionary with correspondence epoch->lr.
        Example: {2: 10**-5, 20: 10**-8}
    :type lr_scheduler: Union[str, Dict[int, float]]
    :param save_best_only: Flag for saving only best model and not every epoch.
    :type save_best_only: bool
    :param old_version: Flag for old version of Unet with a lot of parameters.
    :type old_version: bool
    :param old_upgrade: Flag for upgraded version of old Unet.
    :type old_version: bool
    :param test_as_val: Add test dataset to see its metrics.
    :type test_as_val: Planck_Dataset
    """

    def __init__(self, model_path: str, input_shape: Tuple[int] = (64, 64, 6), n_filters: int = 8,
                 n_blocks: int = 5, n_classes: int = 1, lr: float = 1e-4,
                 add_batch_norm: bool = False, dropout_rate: float = 0.2, weights: str = None,
                 lr_scheduler: Union[str, Dict[int, float]] = None, save_best_only: bool = False,
                 old_version: bool = False, old_upgrade: bool = False,
                 test_as_val: Planck_Dataset = None):
        """Initialize."""
        if old_version:
            self.model = Unet_model_old(input_shape, n_filters, n_blocks, n_classes, lr,
                                        add_batch_norm, dropout_rate, weights, upgrade=old_upgrade)
        else:
            self.model = Unet_model(input_shape, n_classes=n_classes, dropout_rate=dropout_rate,
                                    n_filters=n_filters, n_blocks=n_blocks)
        self.callbacks = [ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                                          save_best_only=save_best_only, mode='min',
                                          save_weights_only=False)]
        if type(lr_scheduler) == str:
            if lr_scheduler == "default":
                self.callbacks.append(LearningRateScheduler(default_lr))
            else:
                print("LR preset not understood. No scheduler.")
        elif type(lr_scheduler) == dict:
            self.callbacks.append(LearningRateScheduler(lambda epoch, lr: dict_lr(epoch, lr,
                                                                                  lr_scheduler)))

        self.test_as_val = False
        if test_as_val is not None:
            self.callbacks.append(AdditionalValidationSets([(test_as_val, "test")], verbose=1))
            self.test_as_val = True
        self.model_path = model_path
        self.history = []

    def get_old_history(self):
        """Load old history."""
        history = pd.read_csv(os.path.join(os.path.dirname(self.model_path), "history.csv"))
        history.drop(columns=["epoch"], inplace=True)
        self.history = history.to_dict("records")

    def train(self, trainset: Planck_Dataset, valset: Planck_Dataset, n_epochs: int,
              init_epoch: int = 0, continue_train: bool = False) -> None:
        """Train model.

        :param trainset: Dataset for training.
        :type trainset: Planck_Dataset
        :param valset: Dataset for validation.
        :type valset: Planck_Dataset
        :param n_epochs: Number of epochs.
        :type n_epochs: int
        :param init_epoch: Index of initial epoch.
        :type init_epoch: int
        :param continue_train: Flag for continuing training.
        :type continuing: bool
        :rtype: None
        """
        if continue_train:
            self.get_old_history()
            init_epoch = len(self.history)

        for i in range(init_epoch, init_epoch + n_epochs):
            print(f"Epoch #{i}")
            history = self.model.fit(trainset.generator(), epochs=i+1, verbose=1,
                                     callbacks=self.callbacks,
                                     validation_data=valset.generator(), initial_epoch=i)
            if self.test_as_val:
                for callback in self.callbacks:
                    if type(callback) == AdditionalValidationSets:
                        self.history.append(callback.history)
                        break
            else:
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


def dict_lr(epoch: int, lr: float, epoch_dict: Dict[int, float]) -> float:
    """LR scheduler depending on epoch_dict.

    :param epoch: Epoch.
    :type epoch: int
    :param lr: Learning rate.
    :type lr: float
    :param epoch_dict: Dictionary with correspondence epoch->lr.
      Example: {2: 10**-5, 20: 10**-8}
    :type epoch_dict: Dict[int, float]
    :rtype: float
    """
    if epoch not in epoch_dict:
        return lr
    return epoch_dict[epoch]


def default_lr(epoch: int, lr: float) -> float:
    """Define changing LR. Default LR scheduler.

    :param epoch: Epoch.
    :type epoch: int
    :param lr: Learning Rate.
    :type lr: float
    :rtype: float
    """
    if epoch % 20 == 0:
        return lr * 0.1
    return lr


def conv_block(inputs: Layer, use_batch_norm: bool = False, dropout_rate: float = 0.3,
               filters: int = 16, kernel_size: Tuple[int] = (3, 3), activation: str = "relu",
               kernel_initializer: str = "he_normal", padding: str = "same"):
    """Double convolution block.

    :param inputs: Input layer.
    :type inputs: Layer
    :param use_batch_norm: Flag for batch normalization.
    :type use_batch_norm: bool
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param filters: Number of filters for convolutions.
    :type filters: int
    :param kernel_size: Kernel size.
    :type kernel_size: Tuple[int]
    :param activation: Activation type.
    :type activation: str
    :param kernel_initializer: Kernel initializer type.
    :type kernel_initializer: str
    :param padding: Padding type.
    :type padding: str
    """

    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
               padding=padding, use_bias=not use_batch_norm)(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout_rate > 0.0:
        c = Dropout(dropout_rate)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer,
               padding=padding, use_bias=not use_batch_norm)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def Unet_model(input_shape: Tuple[int] = (64, 64, 6), n_classes: int = 1,
               dropout_rate: float = 0.3, n_filters: int = 64, n_blocks: int = 4,
               output_activation: str = 'sigmoid', lr: float = 10**-4, weights: str = None
               ) -> Model:
    """Unet model.

    :param input_shape: Shape of input data.
    :type input_shape: Tuple[int]
    :param num_classes: Number of classes.
    :type num_classes: int
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param n_filters: Number of filters.
    :type n_filters: int
    :param n_blocks: Number of blocks.
    :type n_blocks: int
    :param output_activation: Output activation type.
    :type output_activation: str
    :param lr: Learning rate.
    :type lr: float
    :param weights: Path to pretrained weights.
    :type weights: str
    :rtype: Model
    """
    if weights is not None:
        model = load_model(weights, custom_objects={'iou': iou, 'dice': dice})
        model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy',
                                                                                iou, dice])
        return model

    inputs = Input(input_shape)
    x = inputs

    encoder = []
    for i in range(n_blocks):
        x = conv_block(inputs=x, filters=n_filters, use_batch_norm=False, dropout_rate=0.0,
                       padding='same')
        encoder.append(x)
        x = MaxPooling2D((2, 2), strides=2)(x)
        n_filters *= 2

    x = Dropout(dropout_rate)(x)
    x = conv_block(inputs=x, filters=n_filters, use_batch_norm=False, dropout_rate=0.0,
                   padding='same')

    for conv in reversed(encoder):
        n_filters //= 2
        x = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(x)

        x = concatenate([x, conv])
        x = conv_block(inputs=x, filters=n_filters, use_batch_norm=False, dropout_rate=0.0,
                       padding='same')

    outputs = Conv2D(n_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy', iou, dice])
    return model


def Unet_model_old(input_shape: Tuple[int] = (64, 64, 6), n_filters: int = 8, n_blocks: int = 5,
                   n_output_layers: int = 1, lr: float = 1e-4, add_batch_norm: bool = False,
                   dropout_prm: float = 0.2, weights: str = None, upgrade: bool = False) -> Model:
    """Create tensorflow model Unet (old version with big amount of parameters).

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
    :param upgrade: Flag for upgraded version of Unet.
    :type upgrade: bool
    :rtype: Model
    """
    if weights is not None:
        model = load_model(weights, custom_objects={'iou': iou, 'dice': dice})
        model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy',
                                                                                iou, dice])
        return model

    encoder = []
    inputs = Input(input_shape)
    cur = inputs
    for i in range(n_blocks):
        cur = conv_block(inputs=cur, filters=n_filters, use_batch_norm=add_batch_norm,
                         dropout_rate=0.0, padding="same")

        encoder.append(cur)

        cur = MaxPooling2D(padding='valid')(cur)
        cur = Dropout(dropout_prm)(cur)

        n_filters *= 2
    if upgrade:
        cur = conv_block(inputs=cur, filters=n_filters, use_batch_norm=add_batch_norm,
                         dropout_rate=0.0, padding="same")

    for i in reversed(range(n_blocks)):
        cur = UpSampling2D()(cur)
        cur = Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal", padding="same",
                     use_bias=not add_batch_norm)(cur)
        cur = concatenate([cur, encoder[i]], axis=3)

        cur = Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu",
                     kernel_initializer="he_normal", padding="same",
                     use_bias=not add_batch_norm)(cur)
        cur = Dropout(dropout_prm)(cur)

        n_filters //= 2

    cur = Conv2D(n_output_layers, kernel_size=3, padding='same', activation="sigmoid")(cur)

    model = Model(inputs=inputs, outputs=cur)
    model.compile(optimizer=Adam(lr=lr), loss=binary_crossentropy, metrics=['accuracy', iou, dice])
    return model


class AdditionalValidationSets(Callback):
    """
    Edited version form `here
    https://github.com/LucaCappelletti94/keras_validation_sets/blob/master/additional_validation_sets.py
    `_.
    """
    def __init__(self, validation_sets: List[Tuple[Planck_Dataset, str]], verbose: int = 0):
        """Initialize.

        :param validation_sets: list of 2-tuples (dataset, name).
        :type validation_sets: List[Tuple[Planck_Dataset, str]]
        :param verbose: Verbosity mode, 1 or 0.
        :type verbose: int
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) != 2:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose

    def on_train_begin(self, logs: Dict = None):
        """on_train_begin.

        :param logs: Logs
        :type logs: Dict
        """
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """on_epoch_end.

        :param epoch: Number of epoch.
        :type epoch: int
        :param logs: Logs.
        :type logs: Dict
        """
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                dataset, validation_set_name = validation_set
                validation_generator = dataset.generator()
                validation_steps = len(dataset)
            else:
                raise ValueError()
            results = self.model.evaluate(validation_generator, steps=validation_steps)

            for metric, result in zip(self.model.metrics_names, results):
                valuename = validation_set_name + '_' + metric
                self.history.setdefault(valuename, []).append(result)
