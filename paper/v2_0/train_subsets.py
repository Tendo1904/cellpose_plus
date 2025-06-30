"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, argparse
from tifffile import imread, imsave
import numpy as np
from glob import glob
from cellpose import models
from cellpose.io import logger_setup
from cellpose import metrics
from datasets import *


def train_model(
    root,
    model_type="cellpose",
    cell_type=None,
    tissue_type=None,
    platform_type=None,
    full_img=False,
    ntrain=10,
    ntest=0,
    weight_decay=1e-4,
    n_epochs=250,
    learning_rate=0.1,
    batch_size=8,
    nimg_per_epoch=8,
    start_pretrained=False,
    run_eval=True,
    save_results=True,
    save_test_masks=False,
    seed=1,
    flow_threshold=0.4,
    pretrained_model=None,
):
    """
    Train a model for image analysis using specified parameters.

    This method initializes and trains a model, either 'cellpose' or 'Unet',
    based on provided input data and training configurations. It also optionally
    evaluates the model and saves the results.

    Args:
        root: The directory where the training data is located.
        model_type: The type of model to train ('cellpose' or 'Unet').
        cell_type: The specific type of cell to analyze; if None, all types are used.
        tissue_type: The specific type of tissue to analyze; if None, all types are used.
        platform_type: The type of platform used; if None, defaults to available platforms.
        full_img: Indicates whether to use the full image for training.
        ntrain: The number of training images to use.
        ntest: The number of test images to use.
        weight_decay: The weight decay (L2 regularization) parameter for training.
        n_epochs: The total number of epochs to train the model.
        learning_rate: The initial learning rate for the optimizer.
        batch_size: The number of samples per batch during training.
        nimg_per_epoch: The number of images to process per epoch.
        start_pretrained: Flag to indicate whether to start training from a pretrained model.
        run_eval: Flag to indicate whether to run evaluation after training.
        save_results: Flag to indicate whether to save the training results.
        save_test_masks: Flag to indicate whether to save the test masks.
        seed: The random seed for reproducibility.
        flow_threshold: The threshold for flow-based analysis.
        pretrained_model: The path to a pretrained model, if any.

    Returns:
        The path to the trained model.
    """

    if model_type == "cellpose":
        netstr = "cytotorch_0" if start_pretrained else "scratch"
        if pretrained_model is not None:
            pretrained_model = (
                os.fspath(models.MODEL_DIR.joinpath(netstr))
                if start_pretrained
                else None
            )
        model = models.CellposeModel(pretrained_model=pretrained_model, gpu=True)
    else:
        netstr = "cytounettorch_0" if start_pretrained else "scratchunet"
        if pretrained_model is not None:
            pretrained_model = (
                os.fspath(models.MODEL_DIR.joinpath(netstr))
                if start_pretrained
                else None
            )
            netstr = os.path.split(pretrained_model)[-1]
        model = models.UnetModel(pretrained_model=pretrained_model, gpu=True)

    nmasks = 0
    k = 0
    while nmasks == 0:
        train_files, channels, netstrf = get_train_files(
            root,
            cell_type,
            tissue_type,
            platform_type,
            ntrain=ntrain,
            full_img=full_img,
            seed=seed,
        )
        train_data, train_masks = load_data_masks(
            train_files, frac=ntrain if ntrain < 1 else 1.0
        )
        nmasks = np.array([len(np.unique(tl)) - 1 for tl in train_masks]).sum()
        if nmasks == 0:
            np.random.seed(k)
            seed = np.random.randint(100) + 10
            k += 1
    netstr += netstrf
    print(
        f"seed = {seed}, nmasks = {nmasks}, train_files[0] = {os.path.split(train_files[0])[-1]}"
    )
    print(train_data[0].shape, train_masks[0].shape)
    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(n_epochs - 10))
    # for i in range(10):
    #    LR = np.append(LR, LR[-1]/2 * np.ones(2))
    for i in range(10):
        LR = np.append(LR, LR[-1] / 2 * np.ones(5))
    n_epochs = len(LR)

    import torch

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # train model
    model_path = model.train(
        train_data,
        train_masks,
        train_files=None,
        channels=channels,
        normalize=True,
        save_path=root,
        save_every=n_epochs,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        learning_rate=LR,
        momentum=0.9,
        batch_size=batch_size,
        nimg_per_epoch=nimg_per_epoch,
        min_train_masks=50,
        netstr=netstr,
    )
    del model
    torch.cuda.empty_cache()
    if run_eval:
        from cellpose import utils

        diam_train = np.array(
            [utils.diameters(train_masks[k])[0] for k in range(len(train_masks))]
        )
        diam_train[diam_train < 5] = 5.0
        diam_mean = diam_train.mean()

        eval_model(
            model_path,
            diam_mean,
            nmasks,
            root,
            model_type=model_type,
            cell_type=cell_type,
            tissue_type=tissue_type,
            platform_type=platform_type,
            ntest=ntest,
            save_results=save_results,
            save_test_masks=save_test_masks,
            flow_threshold=flow_threshold,
        )

    return model_path


def eval_model(
    model_path,
    diam_mean,
    nmasks,
    root,
    model_type="cellpose",
    cell_type=None,
    tissue_type=None,
    platform_type=None,
    ntest=0,
    flow_threshold=0.4,
    save_results=False,
    save_test_masks=False,
):
    """
    Evaluate a deep learning model on a set of test data and obtain performance metrics.

    This method loads test data and masks, runs the specified model on the test data,
    and computes average precision as well as true positive, false positive, and false
    negative metrics. Optionally, it can save the evaluation results and test masks to files.

    Args:
        model_path: Path to the pre-trained model to be evaluated.
        diam_mean: Average diameter for rescaling the test data.
        nmasks: Number of training masks used for reference in the output.
        root: Root directory for the test data.
        model_type: Type of model to evaluate, defaults to 'cellpose'.
        cell_type: Specific cell type for testing, if applicable.
        tissue_type: Specific tissue type for testing, if applicable.
        platform_type: Specific platform type for testing, if applicable.
        ntest: Number of test files to evaluate, defaults to 0.
        flow_threshold: Threshold for flow decisions in model evaluation, defaults to 0.4.
        save_results: Flag to indicate if results should be saved, defaults to False.
        save_test_masks: Flag to indicate if test masks should be saved, defaults to False.

    Returns:
        Tuple containing test data, test masks, evaluated masks,
        and a tuple of performance metrics (average precision, true positives, false positives, false negatives).
    """
    netstr = os.path.split(model_path)[-1]

    if cell_type is not None:
        test_files = get_livecell_test(root, cell_type, ntest=ntest)
        channels = [0, 0]
        if netstr == "cytotorch_0" or "cytounettorch_0":
            netstr += f"_livecell_{cell_type}"
    else:
        test_files = get_tissuenet_test(root, tissue_type, platform_type, ntest=ntest)
        channels = [2, 1]
        if netstr == "cytotorch_0" or "cytounettorch_0":
            netstr += f"_tissuenet_{tissue_type}_{platform_type}"

    test_data, test_masks = load_data_masks(test_files)

    diam_test = diam_mean * np.ones(len(test_data))
    rescale = 30.0 / diam_test

    if model_type == "cellpose":
        model = models.CellposeModel(pretrained_model=model_path, gpu=True)
        masks, flows, styles = model.eval(
            test_data,
            channels=channels,
            rescale=rescale,
            resample=True,
            net_avg=False,
            batch_size=32,
            tile_overlap=0.1,
            flow_threshold=flow_threshold,
        )
    else:
        model = models.UnetModel(pretrained_model=model_path, gpu=True)
        masks, flows, styles = model.eval(
            test_data, channels=channels, rescale=rescale, net_avg=False, batch_size=32
        )

    threshold = np.arange(0.5, 1.0, 0.05)
    ap, tp, fp, fn = metrics.average_precision(test_masks, masks, threshold=threshold)
    print(ap[:, [0, 5, 8]].mean(axis=0))
    if save_results:
        np.save(
            os.path.join(root, f"models/{netstr}_AP_TP_FP_FN.npy"),
            {
                "threshold": threshold,
                "ap": ap,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ntrain_masks": nmasks,
                "test_files": test_files,
                "diam_mean": diam_mean,
            },
        )
    if save_test_masks:
        for i in range(len(test_files)):
            fname = os.path.splitext(test_files[i])[0]
            imsave(fname + "_" + netstr + "_masks.tif", masks[i])
    return test_data, test_masks, masks, (ap, tp, fp, fn)


def eval_model_cyto(
    root,
    model_type="cellpose",
    cell_type=None,
    tissue_type=None,
    platform_type=None,
    ntest=0,
    flow_threshold=0.4,
    save_results=False,
    save_test_masks=False,
):
    """
    Evaluates a specified model for cell segmentation using provided training data.

    This method loads the necessary training files and their corresponding masks, computes the mean diameter of the training masks, and then calls the underlying model evaluation function with the computed parameters.

    Args:
        root: The root directory containing the training files.
        model_type: The type of model to use for evaluation (default is 'cellpose').
        cell_type: The specific type of cell to evaluate (optional).
        tissue_type: The type of tissue used for evaluation (optional).
        platform_type: The platform on which the model is running (optional).
        ntest: The number of test samples (default is 0).
        flow_threshold: The threshold for flow (default is 0.4).
        save_results: A flag indicating whether to save the evaluation results (default is False).
        save_test_masks: A flag indicating whether to save test masks (default is False).

    Returns:
        None: This method does not return any value.
    """
    if model_type == "cellpose":
        model_path = os.fspath(models.MODEL_DIR.joinpath("cytotorch_0"))
    else:
        model_path = os.fspath(models.MODEL_DIR.joinpath("cytounettorch_0"))
    train_files, channels, netstrf = get_train_files(
        root,
        cell_type=cell_type,
        tissue_type=tissue_type,
        platform_type=platform_type,
        ntrain=0,
    )
    if len(train_files) > 0:
        train_data, train_masks = load_data_masks(train_files)
        from cellpose import utils

        diam_train = np.array(
            [utils.diameters(train_masks[k])[0] for k in range(len(train_masks))]
        )
        diam_train[diam_train < 5] = 5.0
        diam_mean = diam_train.mean()
        print(diam_mean)
        eval_model(
            model_path,
            diam_mean,
            0,
            root,
            model_type=model_type,
            cell_type=cell_type,
            tissue_type=tissue_type,
            platform_type=platform_type,
            save_results=save_results,
            flow_threshold=flow_threshold,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training parameters")

    # settings for locating and formatting images
    parser.add_argument(
        "--dir",
        default=[],
        type=str,
        help="folder containing data to train and test on.",
    )
    parser.add_argument("--train", default=1, type=int, help="train model")
    parser.add_argument(
        "--start_pretrained",
        default=0,
        type=int,
        help="start training from pretrained model",
    )
    parser.add_argument(
        "--model_type",
        default="cellpose",
        type=str,
        help=" model type (cellpose or unet)",
    )
    parser.add_argument(
        "--cell_type", default=None, type=str, help="livecell cell_type"
    )
    parser.add_argument(
        "--tissue_type", default=None, type=str, help="tissuenet tissue_type"
    )
    parser.add_argument(
        "--platform_type", default=None, type=str, help="tissuenet platform_type"
    )
    parser.add_argument("--seed", default=1, type=int, help="random seed for train set")
    parser.add_argument(
        "--ntrain",
        default=10,
        type=float,
        help="# of images in train set, if 0 uses all images",
    )
    parser.add_argument("--ntest", default=50, type=int, help="# of images in test set")
    parser.add_argument(
        "--flow_threshold", default=0.4, type=float, help="flow_threshold"
    )
    parser.add_argument("--full", default=0, type=int, help="use full imgs")

    args = parser.parse_args()
    logger, log_path = logger_setup()

    if args.train:
        train_model(
            args.dir,
            model_type=args.model_type,
            cell_type=args.cell_type,
            tissue_type=args.tissue_type,
            platform_type=args.platform_type,
            start_pretrained=args.start_pretrained,
            seed=args.seed,
            ntrain=args.ntrain,
            ntest=args.ntest,
            full_img=args.full,
            weight_decay=1e-4,
            n_epochs=250,
            learning_rate=0.1,
            batch_size=8,
            nimg_per_epoch=8,
            run_eval=True,
            save_test_masks=False,
            flow_threshold=args.flow_threshold,
        )
    else:
        if args.tissue_type == "loop":
            train_files = glob(os.path.join(root, "train_full/*.tif"))
            train_files = [tf for tf in train_files if tf[-10:] != "_masks.tif"]
            train_types = np.array(
                [os.path.split(tf)[-1].split("_")[0] for tf in train_files]
            )
            train_platforms = np.array(
                [os.path.split(tf)[-1].split("_")[1] for tf in train_files]
            )
            for tissue_type in np.unique(train_types):
                for platform_type in np.unique(platform_types):
                    eval_model_cyto(
                        args.dir,
                        model_type=args.model_type,
                        cell_type=args.cell_type,
                        tissue_type=args.tissue_type,
                        platform_type=args.platform_type,
                        save_results=True,
                        flow_threshold=args.flow_threshold,
                    )

        else:
            eval_model_cyto(
                args.dir,
                model_type=args.model_type,
                cell_type=args.cell_type,
                tissue_type=args.tissue_type,
                platform_type=args.platform_type,
                save_results=True,
                flow_threshold=args.flow_threshold,
            )
