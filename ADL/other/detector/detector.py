"""Module with functions for Planck detector."""
import numpy as np
import healpy as hp
import pandas as pd
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
from ADL.model import Unet_model
from ADL.preproc import one_pixel_fragmentation, pix2radec
from skimage.segmentation import flood
from skimage.filters import roberts
from skimage.measure import moments


def connect_masks(masks: List[np.ndarray], pic_idx: List[Tuple[int]], patch_size: int = 64,
                  big_shape: Tuple[int] = (1024, 1024, 1), data_type: type = np.float64
                  ) -> np.ndarray:
    """Create one big mask from a lot of small ones.

    :param masks: Small masks.
    :type masks: List[np.ndarray]
    :param pic_idx: Indexes for masks.
    :type pic_idx: List[Tuple[int]]
    :param patch_size: Size of patch.
    :type patch_size: int
    :param big_shape: Size of big mask.
    :type big_shape: Tuple[int]
    :param data_type: Data type of new mask.
    :type data_type: type
    :rtype: np.ndarray
    """
    connected_masks = np.zeros(big_shape, dtype=data_type)
    coef = np.zeros(big_shape, dtype=data_type)

    for i in range(len(masks)):
        x, y = pic_idx[i]
        connected_masks[x:x+patch_size, y:y+patch_size, :] += masks[i]
        coef[x:x+patch_size, y:y+patch_size, :] += np.ones((patch_size, patch_size, 1),
                                                           dtype=data_type)

    connected_masks /= coef
    return connected_masks


def scan_sky_Planck(data_path: str, out_path: str, model_path: str, step: int = 64,
                    patch_size: int = 64, nside: int = 2, verbose: bool = True) -> None:
    """Scan of all 48 HEALPix pixels with chosen model.

    Each tile is divided into patches, and then small scans connected together.

    :param data_path: Directory with Planck data.
    :type data_path: str
    :param out_path: Output directory for scans.
    :type out_path: str
    :param model: Tenserflow model.
    :type model: Model
    :param step: Step for scan.
    :type step: int
    :param patch_size: Size of patch.
    :type patch_size: int
    :param nside: Nside for big tiles.
    :type nside: int
    :param verbose: Flag for tqdm.
    :type verbose: bool
    :rtype: None
    """
    model = Unet_model(weights=model_path)
    iter_pixels = range(hp.nside2npix(nside))
    if verbose:
        iter_pixels = tqdm(iter_pixels)
    for ipix in iter_pixels:
        big_pic = np.load(os.path.join(data_path, f'{ipix}.npy'))
        pics = []
        pic_idx = []

        starts = []
        for k in range(2):
            x_st = [i for i in range(0, big_pic.shape[k], step)
                    if i + patch_size <= big_pic.shape[k]] + [big_pic.shape[k] - patch_size]
            starts.append(x_st)

        for i in starts[0]:
            for j in starts[1]:
                pic = big_pic[i:i+patch_size, j:j+patch_size, :]
                if pic.shape == (patch_size, patch_size, pic.shape[-1]):
                    pics.append(pic)
                    pic_idx.append((i, j))
        pred = model.predict(np.array(pics), verbose=0)
        pred = connect_masks(pred, pic_idx)
        np.save(os.path.join(out_path, f'{ipix}.npy'), pred)


def fast_skan_sky_Planck(data_path: str, out_path: str, model_path: str, nside: int = 2,
                         verbose: bool = True) -> None:
    """Fast scan of all 48 HEALPix pixels with chosen model.

    Each tile scanned at once.

    :param data_path:
    :type data_path: str
    :param out_path:
    :type out_path: str
    :param model:
    :type model: Model
    :param nside:
    :type nside: int
    :param verbose: Flag for tqdm.
    :type verbose: bool
    :rtype: None
    """
    model = Unet_model(weights=model_path)
    fast_model = Unet_model(input_shape=(1024, 1024, 6))
    fast_model.set_weights(model.get_weights())
    X = [np.load(os.path.join(data_path, f'{ipix}.npy')) for ipix in range(hp.nside2npix(2))]
    pred = fast_model.predict(np.array(X))
    iter_pixels = range(hp.nside2npix(nside))
    if verbose:
        iter_pixels = tqdm(iter_pixels)
    for ipix in iter_pixels:
        np.save(os.path.join(out_path, f"{ipix}.npy"), pred[ipix])


def find_centroid(pic: np.ndarray) -> Tuple[float]:
    """Find centroid for input mask.

    :param pic: Input mask.
    :type pic: np.ndarray
    :rtype: Tuple[float]
    """
    if len(pic.shape) > 2:
        pic = np.copy(pic).reshape(list(pic.shape)[:-1])
    M = moments(pic)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    return centroid


def get_radius(figure: np.ndarray, center: Tuple[int]) -> Dict[str, float]:
    """For all distances from center of object to its edge, find minimum, mean and maximum.

    :param figure: Mask with one figure.
    :type figure: np.ndarray
    :param center: Center of figure.
    :type center: Tuple[int]
    :rtype: Dict[str, float]
    """
    center = np.array(center)
    edge = np.where(roberts(figure) != 0)
    rads = []
    for point in zip(*edge):
        rads.append(np.linalg.norm(center - np.array(point)))
    if len(rads) == 0:
        return {"min_rad": 0, "mean_rad": 0, "max_rad": 0}
    return {"min_rad": min(rads), "mean_rad": np.mean(rads), "max_rad": max(rads)}


def divide_figures(pic: np.ndarray) -> List[np.ndarray]:
    """For mask with several figures, divide them into different masks.

    :param pic: Input mask.
    :type pic: np.ndarray
    :rtype: List[np.ndarray]
    """
    coords = np.array(np.where(pic != 0))
    ans = []
    while coords.shape[1] != 0:
        seed_point = tuple(coords[:, 0])
        figure = flood(pic, seed_point)
        ans.append(figure)
        pic[np.where(figure)] = 0
        coords = np.array(np.where(pic != 0))
    return ans


def find_centers_on_mask(mask: np.ndarray, thr: float) -> pd.DataFrame:
    """In one large mask find all centers of figures.

    Also find parameters for each figure:
    - area
    - min_rad, mean_rad, max_rad
    - min_pred, max_pred

    :param mask: Input mask.
    :type mask: np.ndarray
    :param thr: Threshold for searching figures.
    :type thr: float
    :rtype: pd.DataFrame
    """
    mask_binary = np.copy(mask)
    mask_binary = np.array(mask_binary >= thr, dtype=np.float32)
    figures = divide_figures(mask_binary)
    prms = []
    for figure in figures:
        prm = {}
        f = np.zeros_like(mask)
        f[np.where(figure)] = mask[np.where(figure)]

        centroid = find_centroid(f)
        prm["x"] = int(centroid[1])
        prm["y"] = int(centroid[0])
        prm["area"] = np.count_nonzero(figure)
        rads = get_radius(figure[:, :, 0], centroid)
        prm.update(rads)
        prm["min_pred"] = np.partition(list(set(f.flatten())), 1)[1]
        prm["max_pred"] = f.max()
        prms.append(pd.DataFrame(prm, index=[0]))

    prms = pd.concat(prms, ignore_index=True)
    return prms


def pix_extract_catalog(pred_path: str, ipix: str, thr: float = 0.1) -> pd.DataFrame:
    """Extract catalog with sky coordinates from one pixel.

    :param pred_path: Path to directory with predicted masks.
    :type pred_path: str
    :param ipix: Number of HEALPix pixel.
    :type ipix: str
    :param thr: Threshold for masks.
    :type thr: float
    :rtype: pd.DataFrame
    """
    pred = np.load(os.path.join(pred_path, f"{ipix}.npy"))
    f_matr = one_pixel_fragmentation(2, ipix, 2**11)
    df = find_centers_on_mask(pred, thr)
    if len(pred) > 0:
        pixels = f_matr[np.array(df["y"]), np.array(df["x"])]
        ra, dec = pix2radec(pixels, nside=2**11)
        df["RA"] = ra
        df["DEC"] = dec
    return df


def sky_extract_catalog(pred_path: str, thr: float = 0.1, verbose: bool = True) -> pd.DataFrame:
    """Extract catalog with sky coordinates for all pixels.

    :param pred_path: Path to directory with predicted masks.
    :type pred_path: str
    :param thr: Threshold for masks.
    :type thr: float
    :param verbose: Flag for tqdm.
    :type verbose: bool
    :rtype: pd.DataFrame
    """
    df = []
    iter_pixels = range(hp.nside2npix(2))
    if verbose:
        iter_pixels = tqdm(iter_pixels)
    for ipix in iter_pixels:
        df.append(pix_extract_catalog(pred_path, ipix, thr))
    df = pd.concat(df, ignore_index=True)
    return df
