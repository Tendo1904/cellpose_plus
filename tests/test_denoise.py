from cellpose import io, denoise
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
import numpy as np


def clear_output(data_dir, image_names):
    """
    No valid docstring found.
    """
    data_dir_2D = data_dir.joinpath("2D")
    data_dir_3D = data_dir.joinpath("2D")
    for image_name in image_names:
        if "2D" in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name))
            ext = ".png"
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = ".tif"
        name, ext = os.path.splitext(cached_file)
        for rtype in ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]:
            output = name + f"_{rtype}.tif"
            if os.path.exists(output):
                os.remove(output)


def test_class_2D(data_dir, image_names):
    """
    Tests the 2D image restoration models by applying denoising, deblurring, and upsampling on a specified grayscale image and saving the results.

    Args:
        data_dir: The directory where the image and output will be stored.
        image_names: A list of image names to process.

    Returns:
        None: The function does not return any value but saves the processed images to the specified directory.
    """
    clear_output(data_dir, image_names)
    image_name = "gray_2D.png"
    img = io.imread(str(data_dir.joinpath("2D").joinpath(image_name)))
    model_types = ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]
    chan = [2, 1, 0]
    chan2 = [1, 0, 0]
    diams = [30.0, 30.0, 15.0]
    shapes = [
        (*img.shape[:2], 1),
        (*img.shape[:2], 1),
        (img.shape[0] * 2, img.shape[1] * 2, 1),
    ]
    for m, model_type in enumerate(model_types):
        model = denoise.DenoiseModel(model_type=model_type, chan2=True)
        img_restore = model.eval(img, diameter=diams[m], channels=[chan[m], chan2[m]])
        assert img_restore.shape == shapes[m]
        io.imsave(
            str(data_dir.joinpath("2D").joinpath(f"gray_2D_{model_type}.tif")),
            img_restore,
        )
    clear_output(data_dir, image_names)


def test_dn_cp_class_2D(data_dir, image_names):
    """
    Tests the Cellpose denoising and classifying models on a 2D image.

    This method reads a specified 2D image, applies various Cellpose model types for denoising,
    deblurring, and upsampling, asserts the output shapes, and saves the restored images.

    Args:
        data_dir: The directory containing the images and results.
        image_names: A list of image names to process.

    Returns:
        None: This method does not return a value but saves processed images to the specified directory.
    """
    clear_output(data_dir, image_names)
    image_name = "rgb_2D.png"
    img = io.imread(str(data_dir.joinpath("2D").joinpath(image_name)))
    model_types = ["denoise_cyto3", "deblur_cyto3", "upsample_cyto3"]
    chan = [2, 1, 0]
    chan2 = [1, 0, 0]
    diams = [30.0, 30.0, 15.0]
    shapes = [
        (*img.shape[:2], 2),
        (*img.shape[:2], 1),
        (img.shape[0] * 2, img.shape[1] * 2, 1),
    ]
    for m, model_type in enumerate(model_types):
        model = denoise.CellposeDenoiseModel(
            model_type="cyto3", restore_type=model_type, chan2_restore=True
        )
        masks, flows, styles, img_restore = model.eval(
            img, diameter=diams[m], channels=[chan[m], chan2[m]]
        )
        assert img_restore.shape == shapes[m]
        assert masks.shape == shapes[m][:2]
        io.imsave(
            str(data_dir.joinpath("2D").joinpath(f"rgb_2D_{model_type}.tif")),
            img_restore,
        )
    clear_output(data_dir, image_names)


def test_cli_2D(data_dir, image_names):
    """
    Runs a test for a 2D cell analysis using the Cellpose model.

    This method executes a command to process images in a specified directory using a predetermined model. It handles output clearing and error management during the execution of the Cellpose command.

    Args:
        data_dir: The directory containing the data to be processed.
        image_names: The names of the images to be processed.

    Returns:
        None: This method does not return any value. It prints the command output or raises a ValueError if an error occurs.
    """
    clear_output(data_dir, image_names)
    model_types = ["denoise_cyto3"]
    chan = [2]
    chan2 = [1]
    for m, model_type in enumerate(model_types):
        cmd = (
            "python -m cellpose --dir %s --pretrained_model %s --restore_type %s --chan %d --chan2 %d --chan2_restore --diameter 30"
            % (str(data_dir.joinpath("2D")), "cyto3", model_type, chan[m], chan2[m])
        )
        try:
            cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
            print(cmd_stdout)
        except Exception as e:
            print(e)
            raise ValueError(e)
        clear_output(data_dir, image_names)
