"""Module with functions for calculating metrics for catalogs."""
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Union
from collections.abc import Callable
import ADL.preproc


def match_det_to_true(det_cat: pd.DataFrame, det_cat_sc: SkyCoord, true_cat: pd.DataFrame,
                      true_name: str, match_dist: float, spec_flag: bool = False) -> Dict:
    """Match detected catalog to ground truth catalog & calculate recall.

    Fill 'found' column, if object was found. If spec_flag is True, fill found_[Special_cat]
    column & calculate precision for this ground truth catalog.

    :param det_cat: Detected catalog.
    :type det_cat: pd.DataFrame
    :param det_cat_sc: SkyCoord for detected catalog.
    :type det_cat_sc: SkyCoord
    :param true_cat: Ground truth catalog.
    :type true_cat: pd.DataFrame
    :param true_name: Name of ground truth catalog.
    :type true_name: str
    :param match_dist: Distance for matching.
    :type match_dist: float
    :param spec_flag: Flag for calculating precision.
    :type spec_flag: bool
    :rtype: Dict
    """
    stats = {}
    true_sc = SkyCoord(ra=np.array(true_cat['RA'])*u.degree,
                       dec=np.array(true_cat['DEC'])*u.degree, frame='icrs')
    idx, d2d, _ = true_sc.match_to_catalog_sky(det_cat_sc)
    matched = d2d.degree <= match_dist
    det_cat.loc[idx[matched], 'found'] = True
    n_matched = np.count_nonzero(matched)
    stats[true_name] = n_matched / len(true_cat)
    if spec_flag:
        if "tRA" not in list(det_cat):
            det_cat["tRA"] = 0
            det_cat["tDEC"] = 0
        det_cat["found_" + true_name] = False
        det_cat.loc[idx[matched], 'found_' + true_name] = True
        det_cat.loc[idx[matched], 'tRA'] = np.array(true_cat["RA"].iloc[matched])
        det_cat.loc[idx[matched], 'tDEC'] = np.array(true_cat["DEC"].iloc[matched])

        n_true_matched = np.count_nonzero(det_cat['found_' + true_name])
        stats['precision_' + true_name] = n_true_matched / len(det_cat)
        stats['found_' + true_name] = n_true_matched
    return stats


def do_all_stats(det_cat: pd.DataFrame, true_cats: Dict[str, pd.DataFrame],
                 match_dist: float = 400/(60)**2, spec_precision: List[str] = []
                 ) -> Dict[str, Union[int, float]]:
    """For detected catalog calculate metrics for selected ground truth catalogs.

    Returns dict, where name of the catalog shows recall. Also contains general precision and
    number of found objects.
    For cats in spec_precision calculate separate precision and number of found objects.

    :param det_cat: Detected catalog.
    :type det_cat: pd.DataFrame
    :param true_cats: Dictionary with ground truth catalogs. {<name_of_catalog> : <catalog>}
    :type true_cats: Dict[str, pd.DataFrame]
    :param match_dist: Distance for matching in degrees.
    :type match_dist: float
    :param spec_precision: List of catalogs for which separate precision will be calcilated.
    :type spec_precision: List[str]
    :rtype: Dict[str, Union[int, float]]
    """
    det_cat['found'] = False
    for cat in spec_precision:
        det_cat['found_' + cat] = False
    det_cat_sc = SkyCoord(ra=np.array(det_cat['RA'])*u.degree,
                          dec=np.array(det_cat['DEC'])*u.degree, frame='icrs')
    stats = {}
    for true_name in true_cats:
        if len(true_cats[true_name]) == 0:
            continue
        spec_flag = False
        if true_name in spec_precision:
            spec_flag = True
        cur_stats = match_det_to_true(det_cat, det_cat_sc, true_cats[true_name], true_name,
                                      match_dist, spec_flag)
        stats.update(cur_stats)

    stats['found'] = np.count_nonzero(det_cat['found'])
    stats['precision'] = stats['found'] / len(det_cat)
    stats['all'] = len(det_cat)
    return stats


def cut_cat_by_pix(df: pd.DataFrame, big_pix: List[int]) -> pd.DataFrame:
    """For input catalog remove all objects, that don't belong to big_pix in HEALPix with nside=2.

    Nested scheme.

    :param df: Input catalog.
    :type df: pd.DataFrame
    :param big_pix: List of pixels.
    :type big_pix: List[int]
    :rtype: pd.DataFrame
    """
    pix = ADL.preproc.radec2pix(df['RA'], df['DEC'], 2)
    df = df[np.in1d(pix, big_pix)]
    df.index = np.arange(len(df))
    return df


def cut_cat(df: pd.DataFrame, dict_cut: Dict[str, Callable[[float], bool]] = {},
            big_pix: List[int] = None) -> pd.DataFrame:
    """For input catalog remove all objects, that don't fit conditions.

    :param df: Input catalog.
    :type df: pd.DataFrame
    :param dict_cut: Conditions for catalog parameters.
    :type dict_cut: Dict[str, Callable[[float], bool]]
    :param big_pix: List of pixels with nside=2.
    :type big_pix: List[int]
    :rtype: pd.DataFrame
    """
    if "l" in dict_cut or "b" in dict_cut:
        sc = SkyCoord(ra=np.array(df['RA'])*u.degree,
                      dec=np.array(df['DEC'])*u.degree, frame='icrs')
        df['b'] = sc.galactic.b.degree
        df['l'] = sc.galactic.l.degree
    for prm, func in dict_cut.items():
        if prm in df:
            df = df.loc[map(func, df[prm])]
            df.index = np.arange(len(df))
    if not (big_pix is None):
        df = cut_cat_by_pix(df, big_pix)
    return df


def cats2dict(dir_path: str) -> Dict[str, pd.DataFrame]:
    """Put all catalogs from directory into dictionary.

    :param dir_path: Path to directory.
    :type dir_path: str
    :rtype: Dict[str, pd.DataFrame]
    """
    files = os.listdir(dir_path)
    files = list(filter(lambda x: x.endswith(".csv"), files))
    true_cats = {file[:-4]: pd.read_csv(os.path.join(dir_path, file)) for file in files}
    return true_cats


def stats_with_rules(det_cat: pd.DataFrame, true_cats: List[pd.DataFrame], rules: Dict = {},
                     big_pix: List[int] = None, match_dist: float = 400/(60)**2,
                     spec_precision: List[str] = []):
    """Calculate metrics for detected catalog with selected rules.

    :param det_cat: Detected catalog.
    :type det_cat: pd.DataFrame
    :param true_cats: Dictionary with ground truth catalogs.
    :type true_cats: List[pd.DataFrame]
    :param rules: Rules for filtering catalogs.
    :type rules: Dict
    :param big_pix: List of HEALPix pixels with nside=2.
    :type big_pix: List[int]
    :param match_dist: Match distance.
    :type match_dist: float
    :param spec_precision: List of catalogs for which precision will be calcilated separately.
    :type spec_precision: List[str]
    """
    det_cat = cut_cat(det_cat, rules, big_pix)
    if len(det_cat) == 0:
        return None
    true_cats = {name: cut_cat(cat, rules, big_pix) for name, cat in true_cats.items()}
    stats = do_all_stats(det_cat, true_cats, match_dist=match_dist, spec_precision=spec_precision)
    stats["n_det"] = len(det_cat)
    return stats


def active_learning_cat(det_cat_path: str, true_dir: str, true_names: List[str], match_dist: float
                        ) -> pd.DataFrame:
    """Create catalog for active learning.

    :param det_cat_path:
    :type det_cat_path: str
    :param true_dir:
    :type true_dir: str
    :param true_names:
    :type true_names: List[str]
    :param match_dist:
    :type match_dist: float
    :rtype: pd.DataFrame
    """
    det_cat = pd.read_csv(det_cat_path)
    det_cat["found"] = False
    true_cats = cats2dict(true_dir)
    det_sc = SkyCoord(ra=det_cat["RA"] * u.degree, dec=det_cat["DEC"] * u.degree, frame="icrs")
    for true_name in true_names:
        match_det_to_true(det_cat, det_sc, true_cats[true_name], true_name, match_dist,
                          spec_flag=True)
    det_cat = det_cat[det_cat["found"]]
    det_cat.drop(columns=["RA", "DEC"], inplace=True)
    det_cat.rename({"tRA": "RA", "tDEC": "DEC"}, axis="columns", inplace=True)
    return det_cat
