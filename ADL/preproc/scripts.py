"""Module with scripts for preprocessing data."""
import os
import healpy as hp
import numpy as np
from tqdm import tqdm
from . import (match_channels, fits2df, normalize_asym, one_pixel_fragmentation,
               draw_masks_and_save, generate_patch_coords)
from typing import List


def preproc_HFI_Planck(inpath: str, outpath: str) -> None:
    """Preprocess .fits files of 6 HFI Planck channels.

    Data can be accessed `here
        <https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/>`_.

    :param inpath: Input directory.
    :type inpath: str
    :param outpath: Output directory.
    :type outpath: str
    :rtype: None
    """
    LFI = ["030", "044", "070"]
    HFI = ["100", "143", "217", "353", "545", "857"]
    files_by_ch = match_channels(inpath, LFI + HFI)
    data_by_ch = {ch: fits2df(os.path.join(inpath, file), "I_STOKES")
                  for ch, file in files_by_ch.items()}
    data_by_ch["030"] = normalize_asym(data_by_ch["030"], p=(0.002, 0.05))
    data_by_ch["044"] = normalize_asym(data_by_ch["044"])
    data_by_ch["070"] = normalize_asym(data_by_ch["070"])
    data_by_ch["100"] = normalize_asym(data_by_ch["100"])
    data_by_ch["143"] = normalize_asym(data_by_ch["143"])
    data_by_ch["217"] = normalize_asym(data_by_ch["217"])
    data_by_ch["353"] = normalize_asym(data_by_ch["353"], p=(10**-4, 0.99))
    data_by_ch["545"] = normalize_asym(data_by_ch["545"], p=(10**-5, 0.9))
    data_by_ch["857"] = normalize_asym(data_by_ch["857"], p=(10**-5, 0.9))

    for ch in LFI:
        data = data_by_ch[ch]
        data_by_ch[ch] = hp.ud_grade(data, 2**11, order_in = "nest", order_out="nest")

    os.mkdir(os.path.join(outpath, "hfi"))
    os.mkdir(os.path.join(outpath, "lfi"))
    os.mkdir(os.path.join(outpath, "healpix_orig"))

    for ch in LFI + HFI:
        np.save(os.path.join(outpath, "healpix_orig/{}.npy".format(ch)), data_by_ch[ch])

    for ipix in tqdm(range(hp.nside2npix(2))):
        pix_matr = one_pixel_fragmentation(2, ipix, 2**11)
        img = np.zeros(pix_matr.shape + (9,), dtype=np.float64)
        for i in range(pix_matr.shape[0]):
            for ch_idx, ch in enumerate(LFI + HFI):
                data = data_by_ch[ch]
                img[i, :, ch_idx] = data[pix_matr[i]]
        np.save(os.path.join(outpath, "lfi", '{}.npy'.format(ipix)), img[:,:,:3])
        np.save(os.path.join(outpath, "hfi", '{}.npy'.format(ipix)), img[:,:,3:])
    return


def generate_masks_and_patches_Planck(inpath: str, outpath: str, n_patches: str,
                                      cats_subset: List[str]) -> None:
    """Generate target data.

    :param inpath: Directory with catalogs. Each catalog should have columns: [RA, DEC].
    :type inpath: str
    :param outpath: Output directory.
    :type outpath: str
    :param n_patches: Approximate amount of patches.
    :type n_patches: str
    :param cats_subset: List of catalogs to use to form patches.
    :type cats_subset: List[str]
    :rtype: None
    """
    print("Creating masks.")
    draw_masks_and_save(inpath, outpath)
    print("Generating coordinates for patches.")
    patches = generate_patch_coords(inpath, n_patches=int(n_patches), cats_subset=cats_subset)
    patches.to_csv(os.path.join(outpath, "pc.csv"), index=False)
    print(f"Number of patches generated: {len(patches)}.")
    # TODO automatically generate description (number of patches for each pixel + catalogs)
    return
