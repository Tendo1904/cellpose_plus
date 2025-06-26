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
    Plots a label on the given axes.

        This method places a text label specified by the ltr parameter at a
        predetermined location in the given matplotlib axis. The position
        is adjusted based on the transformation specified by the trans
        parameter. The font size and weight of the label can be customized.

        Args:
            ltr: A list of labels from which the current label is selected.
            il: The index of the current label to plot from the ltr list.
            ax: The matplotlib axes object where the label will be plotted.
            trans: A transformation that specifies how to position the text.
            fs_title: The font size for the label title (default is 20).

        Returns:
            The updated index after plotting the label.
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
