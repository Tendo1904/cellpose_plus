"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
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
    Plots a label on a given axis at a specific location.

    This method adds a text label to the specified axis using the provided transformation. The label is positioned at the top-left corner of the axis and can be customized with the font size.

    Args:
        ltr: A list or array containing labels.
        il: The index of the label to plot from the list.
        ax: The axis on which to plot the label.
        trans: The transformation to apply to the label's position.
        fs_title: The font size of the title (default is 20).

    Returns:
        An updated index for the label.
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


def outlines_img(imgi, maski, color=[1, 0, 0], weight=2):
    """
    Returns an image with outlines drawn based on a binary mask.

    This method takes an input image and a binary mask, then highlights the outlines of the regions defined by the mask in a specified color. The outlines can be thickened based on the weight parameter.

    Args:
        imgi: The input image as a numpy array.
        maski: A binary mask indicating regions for outlining.
        color: An optional list defining the RGB color for the outlines (default is red).
        weight: An optional integer defining the thickness of the outlines (default is 2).

    Returns:
        A numpy array representing the modified image with outlines.
    """
    img = np.tile(np.clip(imgi.copy(), 0, 1)[:, :, np.newaxis], (1, 1, 3))
    out = np.nonzero(utils.masks_to_outlines(maski[1:-1, 1:-1]))
    img[out[0], out[1]] = np.array(color)
    if weight > 1:
        if weight == 2:
            ix, iy = np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        else:
            ix = np.array([-1, 1, 0, 0])
            iy = np.array([0, 0, 1, 1])
        ix, iy = ix.flatten(), iy.flatten()
        for i in range(len(ix)):
            img[out[0] + ix[i], out[1] + iy[i]] = np.array(color)
    return img
