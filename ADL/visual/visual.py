"""Module with functions for visualization."""
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from typing import Tuple, List
from itertools import cycle


def get_ax(rows: int = 1, cols: int = 1, scale: int = 12, shape: Tuple[int] = None
           ) -> matplotlib.axes.Axes:
    """Return a Matplotlib Axes array to be used in all visualizations.

    :param rows: Number of rows.
    :type rows: int
    :param cols: Number of columns.
    :type cols: int
    :param scale: Scale factor.
    :type scale: int
    :param shape: Shape of image.
    :type shape: Tuple[int]
    :rtype: matplotlib.axes.Axes
    """
    if shape is None:
        shape = (scale * cols, scale * rows)
    else:
        shape = (shape[0] * scale, shape[1] * scale)
    f, ax = plt.subplots(rows, cols, figsize=shape)
    return f, ax


def show_history(ax: matplotlib.axes.Axes, path: str, metrics: List[str] = None,
                 epochs: List = None, find_min: str = None, find_max: str = None,
                 datasets: List[str] = []) -> None:
    """Show history for model.

    :param ax: Axes to plot train curve.
    :type ax: matplotlib.axes.Axes
    :param path: Path to history.csv file.
    :type path: str
    :param metrics: Metrics to show.
    :type metrics: List
    :param epochs: Epochs to show in format [min_epoch, max_epoch).
    :type epochs: list
    :param find_min: Name of metric for which minimum would be found.
    :type find_min: str
    :param find_max: Name of metric for which maximum would be found.
    :type find_max: str
    :param datasets: Validation datasets.
    :type datasets: List[str]
    :rtype: None
    """
    if ax is None:
        ax = get_ax()
    colors = 'bgrcmyk'

    df = pd.read_csv(path)

    if epochs is not None:
        df = df[np.in1d(df["epoch"], range(*epochs))]
    epochs = df['epoch']

    if metrics is None:
        metrics = [k for k in list(df) if k != 'epoch']

    for metric, c in zip(metrics, cycle(colors)):
        s, = ax.plot(epochs, df[metric], c=c)
        s.set_label(metric)

        if len(datasets) > 0:
            for dataset, linestyle in zip(datasets, ["--", ":", "-."]):
                cur_metric = dataset + '_' + metric
                if cur_metric in df.columns:
                    s, = ax.plot(epochs, df[cur_metric], c=c, linestyle=linestyle)
                    s.set_label(cur_metric)

    if find_min is not None:
        m = df[find_min].argmin()
        v = ax.axvline(df.loc[m, "epoch"], c='r')
        ep = df["epoch"][m]
        v.set_label(f"Minimum of {find_min} at {ep}")
    if find_max is not None:
        m = df[find_max].argmax()
        v = ax.axvline(df.loc[m, "epoch"], c='r')
        ep = df["epoch"][m]
        v.set_label(f"Maximum of {find_max} at {ep}")

    ax.set_xlabel(path.split("/")[-2])

    ax.set_xticks(epochs[::5])
    ax.grid()
    ax.legend()
