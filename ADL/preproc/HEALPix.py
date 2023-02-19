"""Module for HEALPix functions."""
import numpy as np
import pandas as pd
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
import ADL.other.metr


def recursive_fill(matr: np.ndarray) -> None:
    """Fill matrix recursively to generate correspondence matrix for HEALPix nested scheme.

    :param matr: Input matrix.
    :type matr: np.ndarray
    :rtype: None
    """
    if matr.shape[0] == 1:
        return

    mid = matr.shape[0] // 2
    np.left_shift(matr, 1, out=matr)
    matr[mid:, :] += 1

    np.left_shift(matr, 1, out=matr)
    matr[:, mid:] += 1

    for i in [0, mid]:
        for j in [0, mid]:
            recursive_fill(matr[i:i+mid, j:j+mid])


def one_pixel_fragmentation(o_nside: int, o_pix: int, f_nside: int) -> np.ndarray:
    """Get correspondence matrix between two nsides for chosen pixel. Nested scheme HEALPix.

    :param o_nside: Larger partition nside.
    :type o_nside: int
    :param o_pix: Pixel index for larger nside.
    :type o_pix: int
    :param f_nside: Smaller partition nside.
    :type f_nside: int
    :rtype: np.ndarray
    """
    depth = int(np.log2(f_nside / o_nside))
    m_len = 2 ** depth
    f_matr = np.full((m_len, m_len), o_pix)
    recursive_fill(f_matr)
    return f_matr


def radec2pix(ra: float, dec: float, nside: int, nest: bool = True) -> np.ndarray:
    """Transform RA, Dec coordinates into HEALPix pixel index.

    :param ra: RA value.
    :type ra: float
    :param dec: Dec value.
    :type dec: float
    :param nside: nside parameter for HEALPix.
    :type nside: int
    :param nest: flag for nested scheme.
    :type nest: bool
    :rtype: np.ndarray
    """
    sc = SkyCoord(ra=np.array(ra)*u.degree, dec=np.array(dec)*u.degree, frame='icrs')
    return hp.ang2pix(nside, sc.galactic.l.degree, sc.galactic.b.degree,
                      nest=nest, lonlat=True)


def pix2radec(ipix: int, nside: int, nest: bool = True) -> Tuple[Union[float, List[float]]]:
    """Transform HEALPix pixel index into RA, Dec coordinates.

    :param ipix: Index of pixel.
    :type ipix: int
    :param nside: nside parameter for HEALPix.
    :type nside: int
    :param nest: flag for nested scheme.
    :type nest: bool
    :rtype: Tuple[Optional[float, List[float]]]
    """
    theta, phi = hp.pix2ang(nside, ipix=np.array(ipix), nest=nest, lonlat=True)

    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree


def flat_arr2matr(h_arr: np.ndarray, pix_matr: np.ndarray) -> np.ndarray:
    """Transform flat HEALPix array into 2d matrix with given correspondence matrix.

    Correspondence matrix should be
    created with one_pixel_fragmentation  method.

    :param h_arr: Flat HEALPix array.
    :type h_arr: np.ndarray
    :param pix_matr: Correspondence matrix.
    :type pix_matr: np.ndarray
    :rtype: np.ndarray
    """
    img = np.zeros_like(pix_matr, dtype=h_arr.dtype)
    for i in range(pix_matr.shape[0]):
        img[i] = h_arr[pix_matr[i]]
    return img


def draw_circles(ras: np.ndarray, decs: np.ndarray, radiuses: Union[np.ndarray, float], nside: int,
                 pix_matr: np.ndarray, centers_in_patch: bool = False) -> np.ndarray:
    """For each pair of RA, Dec coordinates, draw circle with given radius in HEALPix projection.

    :param ras: RA coordinates.
    :type ras: np.ndarray
    :param decs: Dec coordinates.
    :type decs: np.ndarray
    :param radiuses: Radiuses in degrees.
    :type radiuses: Union[np.ndarray, float]
    :param nside: nside for HEALPix.
    :type nside: int
    :param pix_matr: Correspondence matrix for HEALPix.
    :type pix_matr: np.ndarray
    :param centers_in_patch: If this flag is true, circles will only be drawn if their radiuses are
        inside given correspondence matrix.
    :type centers_in_patch: bool
    :rtype: np.ndarray
    """
    h_arr = np.zeros(hp.nside2npix(nside), dtype=np.int8)
    if type(radiuses) != np.ndarray:
        radiuses = [radiuses] * len(ras)
    for ra, dec, radius in zip(ras, decs, radiuses):
        sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
        if centers_in_patch:
            cl_pix = hp.vec2pix(nside, *vec, nest=True)
            if not (cl_pix in pix_matr):
                continue
        pix = hp.query_disc(nside, vec, np.radians(radius), nest=True, inclusive=True)
        h_arr[pix] = 1
    pic = flat_arr2matr(h_arr, pix_matr)
    return pic


def draw_dots(ras: np.ndarray, decs: np.ndarray, nside: int, pix_matr: np.ndarray) -> np.ndarray:
    """For each pair of RA, Dec coordinates, draw dot in HEALPix projection.

    :param ras: RA coordinates.
    :type ras: np.ndarray
    :param decs: Dec coordinates.
    :type decs: np.ndarray
    :param nside: nside for HEALPix.
    :type nside: int
    :param pix_matr: Correspondence matrix for HEALPix.
    :type pix_matr: np.ndarray
    :rtype: np.ndarray
    """
    h_arr = np.zeros(hp.nside2npix(nside), dtype=np.int8)
    h_arr[radec2pix(ras, decs, nside)] = 1
    pic = flat_arr2matr(h_arr, pix_matr)
    return pic


def src_on_batch(patch_line: Dict, f_matr: np.ndarray, cats: Dict[str, pd.DataFrame]
                 ) -> Dict[str, pd.DataFrame]:
    """Return cats with objects that are visible in selected patch.

    Also generate x, y coords (within patch) for each object.

    :param patch_line: Line of patch in table.
    :type patch_line: Dict
    :param f_matr: HEALPix correspondence matrix.
    :type f_matr: np.ndarray
    :param cats: Catalogs.
    :type cats: Dict[str, pd.DataFrame]
    :rtype: Dict[str, pd.DataFrame]
    """
    output = {}
    x, y = patch_line["x"], patch_line["y"]
    f_matr = f_matr[x:x+64, y:y+64].copy()
    for name, cat in cats.items():
        cur_cat = cat[cat["pix2"] == patch_line["pix2"]]
        cur_cat = cur_cat[np.in1d(cur_cat["pix11"], f_matr.flatten())]
        cur_cat.index = np.arange(len(cur_cat))
        cur_cat["x"] = 0
        cur_cat["y"] = 0
        for i in range(len(cur_cat)):
            pix = cur_cat.loc[i, "pix11"]
            coords = np.where(f_matr == pix)
            cur_cat.loc[i, "x"] = coords[0][0]
            cur_cat.loc[i, "y"] = coords[1][0]
        output[name] = cur_cat
    return output


def draw_masks_and_save(cluster_cats_path: str, outpath: str, o_nside: int = 2, nside: int = 2**11,
                        radius: float = 5/60) -> None:
    """Draw masks for training.
    Specify "not_cluster" in catalog's name to put its objects on separate mask.

    :param cluster_cats_path: Directory with catalogs.
    :type cluster_cats_path: str
    :param outpath: Path to save masks.
    :type outpath: str
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :param radius: Radius of masks.
    :type radius: float
    :rtype: None
    """
    cats = ADL.other.metr.cats2dict(cluster_cats_path)
    cluster_cats = {name: cat for name, cat in cats.items() if not "not_cluster" in name}
    non_cluster_cats = {name: cat for name, cat in cats.items() if "not_cluster" in name}
    clusters_cat = pd.concat(cluster_cats.values())
    if len(non_cluster_cats) > 0:
        non_cluster_cat = pd.concat(non_cluster_cats.values())
    else:
        non_cluster_cat = None
    for i in tqdm(range(hp.nside2npix(o_nside))):
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        mask = draw_circles(clusters_cat["RA"], clusters_cat["DEC"], radiuses=radius, nside=nside, pix_matr=pix_matr)
        mask = mask.reshape(mask.shape + (1,))

        if non_cluster_cat is not None:
            non_cluster_mask = draw_circles(non_cluster_cat["RA"], non_cluster_cat["DEC"], radiuses=radius, nside=nside, pix_matr=pix_matr)
            non_cluster_mask = non_cluster_mask.reshape(non_cluster_mask.shape + (1,))
            mask = np.dstack([mask, non_cluster_mask])

        np.save(os.path.join(outpath, f"{i}.npy"), mask)


def calculate_n_src(patch_coords: pd.DataFrame, cluster_cat: pd.DataFrame, o_nside: int = 2,
                    nside: int = 2**11, patch_size: int = 64) -> pd.DataFrame:
    """Calculate n_src for patches.

    :param patch_coords: table with patches.
    :type patch_coords: pd.DataFrame
    :param cluster_cat: List of clusters to calculate.
    :type cluster_cat: pd.DataFrame
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :param patch_size: Size of patch.
    :type patch_size: int
    :rtype: pd.DataFrame
    """
    patch_coords["n_src"] = None
    for i in tqdm(range(hp.nside2npix(2))):
        cur_patch_coords = patch_coords[patch_coords["pix2"] == i]
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        pic = draw_dots(cluster_cat["RA"], cluster_cat["DEC"], nside=nside, pix_matr=pix_matr)
        n_src = []
        for x, y in zip(cur_patch_coords["x"], cur_patch_coords["y"]):
            patch_pic = pic[x:x + patch_size, y:y + patch_size]
            if patch_pic.any():
                n_src.append(np.count_nonzero(patch_pic))
        patch_coords.loc[patch_coords["pix2"] == i, "n_src"] = n_src
    return patch_coords


def update_old_dataset(path: str, cats_subset: List[str], patch_size: int = 64, o_nside: int = 2,
                       nside: int = 2**11) -> None:
    """Add n_src to old dataset.

    :param path: Path to dataset.
    :type path: str
    :param cats_subset: Subset of catalogs.
    :type cats_subset: List[str]
    :param patch_size: Size of patch.
    :type patch_size: int
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :rtype: None
    """
    cats = ADL.other.metr.cats2dict(os.path.join(path, "cats"))
    if cats_subset is not None:
        cats = {key: val for key, val in cats.items() if key in cats_subset}

    df = pd.concat(cats.values())
    patch_coords = pd.read_csv(os.path.join(path, "pc.csv"))
    patch_coords = calculate_n_src(patch_coords, df, patch_size=patch_size, o_nside=o_nside,
                                   nside=nside)

    patch_coords.to_csv(os.path.join(path, "pc.csv"), index=False)


def generate_all_patches(cats_path: str, o_nside: int = 2, nside: int = 2**11,
                         patch_size: int = 64, cats_subset: List[str] = None):
    """Generate all patches withoud skipping.

    :param cats_path: Path to directory with catalogs.
    :type cats_path: str
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :param patch_size: Size of patch.
    :type patch_size: int
    :param cats_subset: Subset of catalogs.
    :type cats_subset: List[str]
    """
    cats = ADL.other.metr.cats2dict(cats_path)
    if cats_subset is not None:
        cats = {key: val for key, val in cats.items() if key in cats_subset}

    df = pd.concat(cats.values())
    all_idx = {"x": [], "y": [], "pix2": []}
    for i in tqdm(range(hp.nside2npix(2))):
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        pic = draw_dots(df["RA"], df["DEC"], nside=nside, pix_matr=pix_matr)
        xs, ys = [], []
        for x in range(0, 1024 - patch_size):
            for y in range(0, 1024 - patch_size):
                patch_pic = pic[x:x+patch_size, y:y+patch_size]
                if patch_pic.any():
                    xs.append(x - patch_size / 2)
                    ys.append(y - patch_size / 2)

        all_idx["x"].extend(xs)
        all_idx["y"].extend(ys)
        all_idx["pix2"].extend([i] * len(xs))

    all_idx = pd.DataFrame(all_idx, index=np.arange(len(all_idx["x"])))
    return all_idx


def generate_patch_coords(cats_path: str, n_patches: int = None, cats_subset: List[str] = None,
                          fit_pixels: Dict[str, List[int]] = None, density_cat_path: str = None,
                          patch_size: int = 64
                          ) -> pd.DataFrame:
    """Create list of dots from which patches can be generated.

    Each patch will contain at least one object from chosen catalogs.

    :param cats_path: Path to directory with catalogs.
    :type cats_path: str
    :param n_patches: Number of patches.
    :type n_patches: int
    :param cats_subset: Subset of catalogs.
    :type cats_subset: List[str]
    :param fit_pixels: Train, val & example pixels to fit train & val patches to example
      distribution. Set example to 'flat' to fit to flat distribution.
    :type fit_pixels: Dict[str, List[int]]
    :param density_cat_path: Path to cat which objects would be used for distributions.
    :type density_cat_path: str
    :rtype: pd.DataFrame
    """
    all_idx = generate_all_patches(cats_path, cats_subset=cats_subset, patch_size=patch_size)
    if n_patches is not None and len(all_idx) > n_patches:
        if fit_pixels is None:
            step = len(all_idx) // n_patches
            all_idx = all_idx.loc[::step]
            all_idx.index = np.arange(len(all_idx))
        else:
            if density_cat_path is not None:
                all_idx = calculate_n_src(all_idx, pd.read_csv(density_cat_path))
            train_patches = all_idx[np.in1d(all_idx["pix2"], fit_pixels["train"])].copy()
            val_patches = all_idx[np.in1d(all_idx["pix2"], fit_pixels["val"])].copy()
            n_train = n_patches * len(fit_pixels["train"]) // 48
            n_val = n_patches * len(fit_pixels["val"]) // 48
            if type(fit_pixels["example"]) == list:
                example_patches = all_idx[np.in1d(all_idx["pix2"], fit_pixels["example"])]
                train_patches = fit_patches_to_distribution(example_patches, train_patches, n_train)
                val_patches = fit_patches_to_distribution(example_patches, val_patches, n_val)
            elif fit_pixels["example"] == "flat":
                train_patches = fit_flat(train_patches, n_train)
                val_patches = fit_flat(val_patches, n_val)
            all_idx = pd.concat([train_patches, val_patches])
    return all_idx


def fit_patches_to_distribution(example: pd.DataFrame, unfitted: pd.DataFrame, n_patches: int
                                ) -> pd.DataFrame:
    """Fit unfitted patches to example distribution.

    :param example: Example patches.
    :type example: pd.DataFrame
    :param unfitted: Unfitted patches.
    :type unfitted: pd.DataFrame
    :param n_patches: Nuber of patches that should be in result.
    :type n_patches: int
    :rtype: pd.DataFrame
    """
    m = max(example["n_src"].max(), unfitted["n_src"].max())
    bins = np.arange(1, m + 2)
    example_f, _ = np.histogram(example["n_src"], bins)
    unfit_f, _ = np.histogram(unfitted["n_src"], bins)

    coef = np.nan_to_num(unfit_f / example_f, 300)
    if any(coef == 0):
        example_f *= int(unfit_f.mean())
        coef = np.nan_to_num(unfit_f / example_f, 300)
    example_scale_f = coef * example_f

    for i, n_src in enumerate(bins[:-1]):
        n_extra = int(unfit_f[i] - example_scale_f[i])
        if n_extra > 0:
            idx = unfitted[unfitted["n_src"] == n_src].index
            chosen = np.random.choice(idx, size=n_extra, replace=False)
            unfitted.drop(chosen, axis="rows", inplace=True)

    chosen = np.random.choice(unfitted.index, len(unfitted) - n_patches, replace=False)
    unfitted.drop(chosen, inplace=True)
    return unfitted


def fit_flat(unfitted: pd.DataFrame, n_patches: int) -> pd.DataFrame:
    """Fit to flat distribution (may add repeatitions).

    :param unfitted: Unfitted patches.
    :type unfitted: pd.DataFrame
    :param n_patches: Number of patches in result.
    :type n_patches: int
    :rtype: pd.DataFrame
    """
    m = unfitted["n_src"].max()
    bins = np.arange(1, m + 2)
    unfit_f, _ = np.histogram(unfitted["n_src"], bins)
    example_scale_f = [n_patches // len(unfit_f) + 1] * len(unfit_f)
    for i, n_src in enumerate(bins[:-1]):
        n_extra = int(unfit_f[i] - example_scale_f[i])
        if n_extra > 0:
            idx = unfitted[unfitted["n_src"] == n_src].index
            chosen = np.random.choice(idx, size=n_extra, replace=False)
            unfitted.drop(chosen, axis="rows", inplace=True)
        elif n_extra < 0:
            idx = unfitted[unfitted["n_src"] == n_src].index
            chosen = np.random.choice(idx, size=-n_extra, replace=True)
            chosen = unfitted.loc[chosen].copy()
            unfitted = pd.concat([unfitted, chosen], ignore_index=True)

    chosen = np.random.choice(unfitted.index, len(unfitted) - n_patches, replace=False)
    unfitted.drop(chosen, inplace=True)
    return unfitted
