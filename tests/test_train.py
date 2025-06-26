from cellpose import io, models, train, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
from glob import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_class_train(data_dir):
    """
    Trains a Cellpose model using images and labels from the specified directory.

        This method prepares the training data, trains the Cellpose model, and saves
        the trained model to the designated directory. It removes any existing model
        files before training to ensure a clean training environment.

        Args:
            data_dir: The directory containing the training images and labels,
                       specifically organized in a '2D/train' structure.

        Returns:
            None: This method does not return any value. It saves the trained model to disk
                  and prints the path to the saved model.
    """
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter="_cyto_masks")
    images, labels, image_names, test_images, test_labels, image_names_test = output
    model = models.CellposeModel(pretrained_model=None, diam_mean=30)
    cpmodel_path = train.train_seg(
        model.net,
        images,
        labels,
        train_files=image_names,
        test_data=test_images,
        test_labels=test_labels,
        test_files=image_names_test,
        channels=[2, 1],
        save_path=train_dir,
        n_epochs=3,
    )[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print(">>>> model trained and saved to %s" % cpmodel_path)


def test_cli_train(data_dir):
    """

    Trains a cell segmentation model using the Cellpose framework and the provided data directory.

    This method constructs and executes command-line instructions to train a model on a specific dataset
    located in the 'train' subdirectory of the given data directory. It first removes any existing model files
    in the model directory before initiating the training process. After training, it identifies the
    pretrained model to be used for inference.

    Args:
        data_dir: The directory containing the dataset. This should have a specific structure with a '2D'
                  subdirectory containing a 'train' folder.

    Raises:
        ValueError: If any errors occur during command execution.

    Returns:
        None
    """
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = (
        "python -m cellpose --train --train_size --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None --chan 2 --chan2 1 --diam_mean 40"
        % train_dir
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)

    model_dir = data_dir.joinpath("2D").joinpath("train").joinpath("models")
    print(model_dir)
    pretrained_models = model_dir.glob("*")
    pretrained_models = [os.fspath(pmodel.absolute()) for pmodel in pretrained_models]
    print(pretrained_models)
    pretrained_model = [
        pmodel for pmodel in pretrained_models if pmodel[-9:] != "_size.npy"
    ][0]
    print(pretrained_model)
    cmd = (
        "python -m cellpose --dir %s --pretrained_model %s --chan 2 --chan2 1 --diam_mean 40"
        % (train_dir, pretrained_model)
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_train_pretrained(data_dir):
    """
    Train a pre-trained model using the Cellpose command line interface.

        This method constructs and executes a command to train a pre-trained Cellpose model
        using data located in a specified directory. It prepares the necessary paths for training
        and cleans up any existing model directories before running the training command.

        Args:
            data_dir: The directory containing the training data, expected to have a structure
                      where '2D/train' holds the training images, and '_cyto_masks'
                      holds the corresponding masks.

        Raises:
            ValueError: If an error occurs while executing the training command.

        Returns:
            None
    """
    train_dir = str(data_dir.joinpath("2D").joinpath("train"))
    model_dir = str(data_dir.joinpath("2D").joinpath("train").joinpath("models"))
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = (
        "python -m cellpose --train --train_size --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model cyto --chan 2 --chan2 1"
        % train_dir
    )
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)
