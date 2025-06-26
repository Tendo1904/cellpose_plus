from cellpose import io, models, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import numpy as np


def test_shape_2D():
    """
    Tests the shape of the output masks from the Cellpose model on a 2D image.

        This function creates a zero-initialized image with dimensions 224x1x224,
        initializes a Cellpose model for cytoplasm segmentation, and evaluates the model
        on the created image. It then asserts that the shape of the output masks is
        as expected (224x224).

        Parameters:
          None

        Returns:
          None
    """
    img = np.zeros((224, 1, 224))
    model = models.Cellpose(model_type="cyto3")
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[0, 0], channel_axis=1)
    assert masks.shape == (224, 224)


def test_shape_3D():
    """
    Test the output shape of a 3D image processing model.

        This function initializes a 3D image with specific dimensions,
        applies the Cellpose model to evaluate it, and asserts that the
        output masks have the expected shape corresponding to the number
        of images processed.

        Parameters:
            None

        Returns:
            None
    """
    img = np.zeros((224, 224, 1, 5, 1))
    model = models.Cellpose(model_type="cyto3")
    masks, flows, _, _ = model.eval(
        img, diameter=30, channels=[0, 0], channel_axis=None, z_axis=3
    )
    assert masks.shape == (5, 224, 224)


def test_shape_stitch():
    """
    Test the shape of the output masks from the Cellpose model.

        This method creates a dummy image and evaluates a Cellpose model on it to ensure
        that the shape of the output masks matches the expected dimensions. The purpose of
        this test is to confirm that the model correctly processes the input image without
        any shape discrepancies.

        Returns:
            None: This method does not return any value.
    """
    img = np.zeros((5, 224, 224))
    model = models.Cellpose(model_type="cyto3")
    masks, flows, _, _ = model.eval(
        img, diameter=30, channels=[0, 0], stitch_threshold=0.9
    )
    assert masks.shape == (5, 224, 224)


def test_shape_2D_2chan():
    """
    Test the shape of the output masks from the Cellpose model.

        This method creates a dummy 2D image with 2 channels and evaluates
        the Cellpose model to ensure that the output masks have the expected
        shape.

        Parameters:
            None

        Returns:
            None
    """
    img = np.zeros((224, 3, 224))
    model = models.Cellpose(model_type="cyto3")
    masks, flows, _, _ = model.eval(img, diameter=30, channels=[2, 1], channel_axis=1)
    assert masks.shape == (224, 224)
