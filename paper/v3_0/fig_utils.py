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
    Plots a label on the given axes.

        This method adds a text label from a provided list to the specified
        axes at a designated position determined by transformation parameters.

        Args:
            ltr: A list of labels from which to choose the label to plot.
            il: The index of the current label in the list to be plotted.
            ax: The axes object on which to plot the label.
            trans: Transformation object that determines the position of the label.
            fs_title: The font size of the title. Defaults to 20.

        Returns:
            The updated index for the next label.
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
    Generates an outlined version of an image based on a mask.

        This function takes an input image and a mask, and creates an output image
        where the outlines of the mask are highlighted in a specified color. It can
        also expand the width of the outlines based on the specified weight.

        Args:
            imgi: The input image to be outlined.
            maski: The binary mask used to determine which areas of the image should be outlined.
            color: A list representing the RGB color for the outlines (default is red).
            weight: An integer indicating the thickness of the outlines (default is 2).

        Returns:
            The outlined image with the specified color applied to the areas defined by the mask.
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
