"""Scripts for metrics."""
import numpy as np
import pandas as pd
from . import cats2dict, stats_with_rules
from ADL.model import pixels
from typing import List


pregen_thr = {"brcat": [5.05335105, 5.1069375,  5.1630591, 5.223298, 5.29437,
                        5.3639328, 5.43414885, 5.5157134, 5.60974235, 5.711204,
                        5.8320221, 5.9664762, 6.12062285, 6.2958075, 6.52273275,
                        6.826352, 7.2506403, 8.0152959, 10.0664315, 10.5,
                        12.]}


def calc_prec_recall_by_range_parameter(det_cat_path: str, true_cats_path: str, out_path: str,
                                        rules_preset: str, range_prm: str, range_preset: str,
                                        pixels_preset: str, n_bins: int = 20,
                                        spec_precision: List[str] = ["eROSITA"]) -> None:
    """Create precision-recall .csv file for detected catalog.

    :param det_cat_path: Path to detected catalog.
    :type det_cat_path: str
    :param true_cats_path: Path to directory with ground truth catalogs.
    :type true_cats_path: str
    :param out_path: Output path.
    :type out_path: str
    :param rules_preset: Rules for filtering catalog.
    :type rules_preset: str
    :param range_prm: Name of range parameter.
    :type range_prm: str
    :param range_preset: Preset for range values.
    :type range_preset: str
    :param pixels: Preset for pixels.
    :type pixels: str
    :param n_bins: Number of values.
    :type n_bins: int
    :rtype: None
    """
    true_cats = cats2dict(true_cats_path)
    det_cat = pd.read_csv(det_cat_path)

    rules = {}
    if "b20" in rules_preset:
        rules["b"] = lambda x: abs(x) > 20
    if "east" in rules_preset:
        rules["l"] = lambda x: 0 <= x <= 180
    elif "west" in rules_preset:
        rules["l"] = lambda x: 180 <= x <= 360

    if range_preset == "linear":
        min_val = det_cat[range_prm].min()
        max_val = det_cat[range_prm].max()
        thr_vals = np.arange(min_val, max_val, (max_val - min_val) / n_bins)
    elif range_preset == "range1":
        thr_vals = np.arange(0, 1, 1/n_bins)
    elif range_preset in pregen_thr:
        thr_vals = pregen_thr[range_preset]
    elif range_preset == "quantile":
        thr_vals = [det_cat[range_prm].quantile(i) for i in np.arange(1 / n_bins, 1, 1 / n_bins)]
    else:
        print("Range preset is not recognized.")
        return

    pixels_dir = list(filter(lambda x: not x.startswith("_"), dir(pixels)))
    if pixels_preset in pixels_dir:
        selected_pix = getattr(pixels, pixels_preset)
    elif pixels_preset == "all":
        selected_pix = None
    else:
        print("Pixels parameter is not recognized.")
        return

    stats_df = []
    for thr in thr_vals:
        rules[range_prm] = lambda x: x > thr
        stats = stats_with_rules(det_cat, true_cats, rules, spec_precision=spec_precision,
                                 big_pix=selected_pix)
        if stats is not None:
            stats[range_prm] = thr
            stats_df.append(pd.DataFrame(stats, index=[0]))
    stats_df = pd.concat(stats_df, ignore_index=True)
    stats_df.to_csv(out_path, index=False)
