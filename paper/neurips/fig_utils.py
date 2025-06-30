"""
Copyright © 2024 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from cellpose import utils

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


def plot_label(ltr, il, ax, trans, fs_title=20):
    """
    Plots a label at the top-left corner of a given axis.

    This method places a text label specified by the `ltr` list at the position (0.0, 1.0)
    on the provided axes object, applying the specified transformations and formatting.

    Args:
        ltr: A list of labels from which to select the text to plot.
        il: The index of the label in the `ltr` list to be plotted.
        ax: The axes object where the label will be plotted.
        trans: The transformation to be applied to the text position.
        fs_title: The font size for the title text (default is 20).

    Returns:
        The updated index `il`, incremented by 1.
    """
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il
