"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, argparse
from tifffile import imread, imsave
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from cellpose import models
from cellpose.io import logger_setup
from cellpose.transforms import normalize_img
from cellpose import metrics
from tqdm import tqdm, trange
from natsort import natsorted

### FUNCTIONS FOR USING ALL DATASETS TOGETHER -------------- ###


def reshape_and_normalize(img, img_type):
    """
    Reshapes and normalizes an image based on its type.

    This method modifies the input image according to its specified type,
    rearranging its dimensions or stacking additional channels as necessary.
    It also normalizes the image for further processing.

    Args:
        img: The input image that needs to be reshaped and normalized.
        img_type: A tuple indicating the type of the image, which influences
                  how the image will be processed.

    Returns:
        The reshaped and normalized image.
    """
    if img_type[0] == "tn":
        img = img[::-1]
    if img_type[0] == "cp":
        if img_type[1] == "nuclei":
            img = np.stack((img[0], np.zeros_like(img[0])), axis=0)
        elif img_type[1] == "cyto":
            img = img[:, :, [1, 0]].transpose(2, 0, 1)
    elif img.ndim < 3:
        img = np.stack((img, np.zeros_like(img)), axis=0)
    img = normalize_img(img, axis=0)
    return img


def get_all_files(root):
    """
    Retrieves all training and testing files along with their corresponding types from specified datasets.

    This method searches through the given root directory for specific datasets and collects file paths for both training and testing images while filtering out certain undesired files. It organizes the collected files and their types for further processing.

    Args:
        root: The root directory path where the datasets are located.

    Returns:
        A tuple containing:
            - A list of training file paths.
            - A list of training file types.
            - A list of testing file paths.
            - A list of testing file types.
    """
    train_files_all = []
    train_types_all = []
    test_files_all = []
    test_types_all = []

    dsets = ["cellpose_cyto_dataset", "livecell_dataset", "tissuenet_dataset"]
    ctype = ["cyto"]
    ext = [".png", ".tif", ".tif"]
    tfs = ["", "lc", "tn"]
    train_str = ["train", "train", "train_full"]
    for k, dset in enumerate(dsets):
        train_files = glob(os.path.join(root + dset, f"{train_str[k]}/*{ext[k]}"))
        train_files = [
            tf
            for tf in train_files
            if tf[-10:] != f"_masks.png"
            and tf[-10:] != f"_masks.tif"
            and tf[-10:] != "_flows.tif"
        ]
        train_files = natsorted(train_files)
        if k > 0:
            train_types = np.array(
                [[tfs[k], *os.path.split(tf)[-1].split("_")[:2]] for tf in train_files],
                dtype=object,
            )
        else:
            train_types = np.array(
                [["cp", ctype[k]] for i in range(len(train_files))], dtype=object
            )
        train_files_all.extend(train_files)
        train_types_all.extend(train_types)

        test_files = glob(os.path.join(root + dset, f"test/*{ext[k]}"))
        test_files = [
            tf
            for tf in test_files
            if tf[-10:] != f"_masks.png"
            and tf[-10:] != f"_masks.tif"
            and tf[-10:] != "_flows.tif"
        ]

        test_files = natsorted(test_files)
        if k > 0:
            test_types = np.array(
                [[tfs[k], *os.path.split(tf)[-1].split("_")[:2]] for tf in test_files],
                dtype=object,
            )
        else:
            test_types = np.array(
                [["cp", ctype[k]] for i in range(len(test_files))], dtype=object
            )
        test_files_all.extend(test_files)
        test_types_all.extend(test_types)

    train_files = train_files_all
    train_types = train_types_all
    test_files = test_files_all
    test_types = test_types_all

    return train_files, train_types, test_files, test_types


def load_train_test_all(train_files, train_types, test_files, test_types):
    """reshape train and test data with cyto chan 0 and nuclei chan 1"""
    from cellpose.io import imread

    train_data = [
        reshape_and_normalize(imread(train_files[i]), train_types[i])
        for i in trange(len(train_files))
    ]
    test_data = [
        reshape_and_normalize(imread(test_files[i]), test_types[i])
        for i in trange(len(test_files))
    ]
    return train_data, test_data


### FUNCTIONS FOR GETTING TISSUENET FILES + SUBSETS --------------- ###


def get_tissuenet_val(root, tissue_type="ALL", platform_type="ALL", nval=8, seed=1):
    """
    Retrieves validation data from the TissueNet dataset based on specified tissue and platform types.

    This method loads the validation dataset from a specified root directory, filters the data based on the given tissue and platform types, and returns a specified number of validation samples and their corresponding labels.

    Args:
        root: The directory path where the validation dataset is stored.
        tissue_type: The type of tissue to filter the validation data. Default is 'ALL', which means all tissue types will be included.
        platform_type: The platform type to filter the validation data. Default is 'ALL', which means all platform types will be included.
        nval: The number of validation samples to return. Default is 8.
        seed: A seed value for random number generation (if applicable). Default is 1.

    Returns:
        A tuple containing:
            - A NumPy array of validation samples, limited to the specified number.
            - A NumPy array of corresponding labels for the validation samples, structured as required.
    """
    dat = np.load(root + "npz/tissuenet_v1.0_val.npz")
    data = dat["X"]
    labels = dat["y"]
    if tissue_type != "ALL":
        ix = (
            np.logical_and(
                dat["tissue_list"] == tissue_type, dat["platform_list"] == platform_type
            )
        ).nonzero()[0]
    X_val, y_val = data[ix], labels[ix]
    return X_val[:nval], y_val[:nval, :, :, [0]]


def get_tissuenet_train(
    root, tissue_type="ALL", platform_type="ALL", ntrain=10, full_img=False, seed=1
):
    """
    Retrieves a list of training image file paths from the specified root directory based on tissue type and platform type.

    This method filters and randomly selects training images, allowing for the option to retrieve full images or their variations.

    Args:
        root: The root directory containing the training images.
        tissue_type: The type of tissue to filter the images by. Default is 'ALL', which includes all tissue types.
        platform_type: The platform type to filter the images by. Default is 'ALL', which includes all platforms.
        ntrain: The number of training images to return. If set to None or less than 1, all matching images will be returned.
        full_img: A boolean indicating whether to return full images or their variations. Default is False.
        seed: The seed for random number generation to ensure reproducibility. Default is 1.

    Returns:
        A list of file paths to the training images based on the specified criteria.
    """
    train_files = glob(os.path.join(root, "train_full/*.tif"))
    train_files = [
        tf
        for tf in train_files
        if tf[-10:] != "_masks.tif" and tf[-10:] != "_flows.tif"
    ]
    train_types = np.array([os.path.split(tf)[-1].split("_")[0] for tf in train_files])
    train_platforms = np.array(
        [os.path.split(tf)[-1].split("_")[1] for tf in train_files]
    )
    # get indices of files
    if tissue_type == "ALL":
        ix_train = np.arange(0, len(train_files))
    else:
        ix_train = (train_types == tissue_type) * (train_platforms == platform_type)
        ix_train = ix_train.nonzero()[0]

    if ntrain is not None and ntrain > 0:
        np.random.seed(seed)
        iperm_train = np.random.permutation(len(ix_train))
        itrain = ix_train[iperm_train[: int(np.ceil(ntrain))]]
        train_files_full = [train_files[it] for it in itrain]
        if full_img:
            return train_files_full
        else:
            krand = np.random.randint(4, size=(len(ix_train),))
            train_files = []
            for i, tf in enumerate(train_files_full):
                fname = os.path.splitext(os.path.split(tf)[-1])[0]
                fnewname = os.path.join(root, f"train/{fname}_{krand[i]}.tif")
                train_files.append(fnewname)
    else:
        itrain = ix_train
        train_files_full = [train_files[it] for it in itrain]
        if full_img:
            return train_files_full
        else:
            train_files = []
            for i, tf in enumerate(train_files_full):
                for k in range(4):
                    fname = os.path.splitext(os.path.split(tf)[-1])[0]
                    fnewname = os.path.join(root, f"train/{fname}_{k}.tif")
                    train_files.append(fnewname)
    return train_files


def get_tissuenet_test(root, tissue_type="ALL", platform_type="ALL", ntest=0):
    """
    Get a list of test files based on specified tissue and platform types.

    This method retrieves test image files from a specified root directory,
    filtering them based on the desired tissue type and platform type.
    Optionally, it can return a specified number of randomly selected test files.

    Args:
        root: The root directory path where test files are located.
        tissue_type: The type of tissue to filter the test files. If set to 'ALL', no filtering is applied.
        platform_type: The type of platform to filter the test files. If set to 'ALL', no filtering is applied.
        ntest: The number of test files to return. If set to 0 or None, all filtered test files are returned.

    Returns:
        A list of filtered test file paths.
    """
    test_files = glob(os.path.join(root, "test/*.tif"))
    test_files = [
        tf for tf in test_files if tf[-10:] != "_masks.tif" and tf[-10:] != "_flows.tif"
    ]
    test_types = np.array([os.path.split(tf)[-1].split("_")[0] for tf in test_files])
    test_platforms = np.array(
        [os.path.split(tf)[-1].split("_")[1] for tf in test_files]
    )
    if tissue_type == "ALL":
        ix_test = np.arange(0, len(test_files))
    else:
        ix_test = (test_types == tissue_type) * (test_platforms == platform_type)
        ix_test = ix_test.nonzero()[0]

    if ntest is not None and ntest > 0:
        np.random.seed(1)
        iperm_test = np.random.permutation(len(ix_test))
        itest = ix_test[iperm_test[:ntest]]
    else:
        itest = ix_test
    test_files = [test_files[it] for it in itest]
    return test_files


### FUNCTIONS FOR GETTING LIVECELL FILES + SUBSETS --------------- ###


def get_livecell_train(root, cell_type="ALL", ntrain=10, seed=0):
    """
    Fetches a list of training file paths for a specified cell type from a given directory.

    This method retrieves TIF files from a specified training directory while excluding
    mask and flow files. It allows for filtering by cell type and can return a limited
    number of training files based on the specified parameters.

    Args:
        root: The root directory where training files are located.
        cell_type: The type of cell to filter the training files. Defaults to 'ALL'.
        ntrain: The maximum number of training files to return. Defaults to 10.
        seed: The seed for the random number generator to ensure reproducibility. Defaults to 0.

    Returns:
        A list of training file paths matching the specified criteria.
    """
    train_files = glob(os.path.join(root, "train/*.tif"))
    train_files = [
        tf
        for tf in train_files
        if tf[-10:] != "_masks.tif" and tf[-10:] != "_flows.tif"
    ]
    train_types = np.array([os.path.split(tf)[-1].split("_")[0] for tf in train_files])

    # get indices of files
    if cell_type == "ALL":
        ix_train = np.arange(0, len(train_files))
    else:
        ix_train = train_types == cell_type
        ix_train = ix_train.nonzero()[0]
    if ntrain is not None and ntrain > 0:
        np.random.seed(seed)
        iperm_train = np.random.permutation(len(ix_train))
        itrain = ix_train[iperm_train[: int(np.ceil(ntrain))]]
    else:
        itrain = ix_train
    train_files = [train_files[it] for it in itrain]
    return train_files


def get_livecell_test(root, cell_type="ALL", ntest=0):
    """
    Retrieve a list of test image files based on specified criteria.

    This method searches for test image files in a given directory, filters them based on the specified cell type, and optionally limits the number of returned files.

    Args:
        root: The root directory where test image files are located.
        cell_type: The type of cells to filter the test files by. Defaults to 'ALL'.
        ntest: The maximum number of test files to return. If set to 0 or None, returns all matching files.

    Returns:
        A list of test image file paths that match the specified conditions.
    """
    test_files = glob(os.path.join(root, "test/*.tif"))
    test_files = [
        tf for tf in test_files if tf[-10:] != "_masks.tif" and tf[-10:] != "_flows.tif"
    ]
    test_types = np.array([os.path.split(tf)[-1].split("_")[0] for tf in test_files])

    if cell_type == "ALL":
        ix_test = np.arange(0, len(test_files))
    else:
        ix_test = test_types == cell_type
        ix_test = ix_test.nonzero()[0]
    if ntest is not None and ntest > 0:
        np.random.seed(1)
        iperm_test = np.random.permutation(len(ix_test))
        itest = ix_test[iperm_test[:ntest]]
    else:
        itest = ix_test
    test_files = [test_files[it] for it in itest]
    return test_files


### FUNCTIONS FOR LIVECELL/TISSUENET LOADING ------------------------ ###


def load_data_masks(files, frac=1.0):
    """
    Loads image data and their corresponding masks from specified files.

    Args:
        files: A list of file paths to load images from.
        frac: A fraction determining the portion of the data to load (default is 1.0).
              If set to 0.5, loads half of the last dimension.
              If set to 0.25, loads half of the last two dimensions.

    Returns:
        A tuple containing:
            - data: A list of loaded image data.
            - masks: A list of loaded masks corresponding to the images.
    """
    data = [imread(file) for file in files]
    masks = [
        imread(os.path.splitext(file)[0] + "_masks.tif").astype(np.uint16)
        for file in files
    ]
    if frac == 0.5:
        data = [d[..., : d.shape[-1] // 2] for d in data]
        masks = [m[..., : m.shape[-1] // 2] for m in masks]
    elif frac == 0.25:
        data = [d[..., : d.shape[-2] // 2, : d.shape[-1] // 2] for d in data]
        masks = [m[..., : m.shape[-2] // 2, : m.shape[-1] // 2] for m in masks]
    return data, masks


def get_train_files(
    root, cell_type, tissue_type, platform_type, ntrain=10, full_img=False, seed=1
):
    """
    Gets training files and associated metadata based on provided specifications.

    This method retrieves training files for either live cell data or tissue data, depending on the specified cell type or tissue type. It also features options for the number of training samples to retrieve, whether to use full images, and a seed for randomization.

    Args:
        root: The root directory where the training files are located.
        cell_type: The type of cell for which to fetch training files; should be specified if retrieving live cell data.
        tissue_type: The type of tissue for which to fetch training files; should be specified if retrieving tissue data.
        platform_type: The platform type for the tissue data.
        ntrain: The number of training samples to retrieve (default is 10).
        full_img: A boolean indicating if full images should be used (default is False).
        seed: An integer seed for randomization (default is 1).

    Returns:
        A tuple containing:
            - A list of training file paths.
            - A list of channel indices used for the training.
            - A string representing the network string format.
    """
    if cell_type is not None:
        train_files = get_livecell_train(root, cell_type, ntrain=ntrain, seed=seed)
        channels = [0, 0]
        netstrf = f"_livecell_{cell_type}_ntrain_{ntrain}_seed_{seed}"
    elif tissue_type is not None:
        train_files = get_tissuenet_train(
            root,
            tissue_type,
            platform_type,
            ntrain=ntrain,
            full_img=full_img,
            seed=seed,
        )
        channels = [2, 1]
        netstrf = (
            f"_tissuenet_{tissue_type}_{platform_type}_ntrain_{ntrain}_seed_{seed}"
        )
    if full_img:
        netstrf += "_FULL"
    return train_files, channels, netstrf


### FUNCTION FOR LIVECELL PREPROCESS ------------------------------ ###


def remove_overlaps(masks, medians, overlap_threshold=0.75):
    """replace overlapping mask pixels with mask id of closest mask
    if mask fully within another mask, remove it
    masks = Nmasks x Ly x Lx
    """
    cellpix = masks.sum(axis=0)
    igood = np.ones(masks.shape[0], "bool")
    for i in masks.sum(axis=(1, 2)).argsort():
        npix = float(masks[i].sum())
        noverlap = float(masks[i][cellpix > 1].sum())
        if noverlap / npix >= overlap_threshold:
            igood[i] = False
            cellpix[masks[i] > 0] -= 1
            # print(cellpix.min())
    print(f"removing {(~igood).sum()} masks")
    masks = masks[igood]
    medians = medians[igood]
    cellpix = masks.sum(axis=0)
    overlaps = np.array(np.nonzero(cellpix > 1.0)).T
    dists = ((overlaps[:, :, np.newaxis] - medians.T) ** 2).sum(axis=1)
    tocell = np.argmin(dists, axis=1)
    masks[:, overlaps[:, 0], overlaps[:, 1]] = 0
    masks[tocell, overlaps[:, 0], overlaps[:, 1]] = 1

    # labels should be 1 to mask.shape[0]
    masks = (
        masks.astype(int)
        * np.arange(1, masks.shape[0] + 1, 1, int)[:, np.newaxis, np.newaxis]
    )
    masks = masks.sum(axis=0)
    return masks


def ann_to_masks(annotations, anns, overlap_threshold=0.75):
    """list of coco-format annotations with masks to single image"""
    masks = []
    k = 0
    medians = []
    for ann in anns:
        mask = annotations.annToMask(ann)
        masks.append(mask)
        ypix, xpix = mask.nonzero()
        medians.append(np.array([ypix.mean(), xpix.mean()]))
        k += 1
    masks = np.array(masks).astype("int")
    medians = np.array(medians)
    masks = remove_overlaps(masks, medians, overlap_threshold=overlap_threshold)
    return masks


def livecell_ann_to_masks(img_dir, annotation_file):
    """
    Converts LiveCell annotations to mask files for images in a given directory.

    This method processes images stored in a specified directory, reading the corresponding
    annotations from a COCO format file. For each image, it generates segmentation masks
    from the annotations and saves them as TIFF files.

    Args:
        img_dir: The directory containing subdirectories of image files organized by class.
        annotation_file: The path to the annotation file in COCO format that provides
                         information about image segmentation.

    Returns:
        None: This method does not return a value. It saves the generated masks as
              TIFF files in the same directory as the input images.
    """
    from pycocotools.coco import COCO
    from tifffile import imsave

    img_dir_classes = glob(img_dir + "*/")
    classes = [img_dir_class.split(os.sep)[-2] for img_dir_class in img_dir_classes]
    print(classes)

    train_files = []
    train_class_files = []
    for cclass, img_dir_class in zip(classes, img_dir_classes):
        train_files.extend(glob(img_dir_class + "*.tif"))
        train_class_files.append(glob(img_dir_class + "*.tif"))

    annotations = COCO(annotation_file)
    imgIds = list(annotations.imgs.keys())

    for train_class_file in train_class_files:
        for i in range(len(train_class_file)):
            filename = train_class_file[i]
            fname = os.path.split(filename)[-1]
            loc = np.array(
                [annotations.imgs[imgId]["file_name"] == fname for imgId in imgIds]
            ).nonzero()[0]
            if len(loc) > 0:
                imgId = imgIds[loc[0]]
                annIds = annotations.getAnnIds(imgIds=[imgId], iscrowd=None)
                anns = annotations.loadAnns(annIds)
                masks = ann_to_masks(annotations, anns, overlap_threshold=0.75)
                masks = masks.astype(np.uint16)
                maskname = os.path.splitext(filename)[0] + "_masks.tif"
                imsave(maskname, masks)
                print(f"saved masks at {maskname}")
