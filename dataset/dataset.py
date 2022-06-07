import numpy as np
from imgaug.augmenters import Augmenter
from typing import Tuple


def do_aug(images: np.ndarray, masks: np.ndarray, augmentation: Augmenter) -> Tuple[np.ndarray]:
    """Apply augmentation to images and masks.

    :param images: [n_batches, size, size, n_channels]-shaped array with images.
    :type images: np.ndarray
    :param masks: [n_batches, size, size, 1]-shaped array with masks.
    :type masks: np.ndarray
    :param augmentation: Augmenter.
    :type augmentation: Augmenter
    :rtype: Tuple[np.ndarray]
    """
    det = augmentation.to_deterministic()
    images = det.augment_images(images.copy())
    masks = det.augment_images(masks.copy())
    return images, masks
