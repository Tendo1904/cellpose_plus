from cellpose import io, models, train, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil
from glob import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_class_train(data_dir):
    """
    Trains a Cellpose model using the provided training data.

    This method prepares the training environment by removing any existing model directories,
    loads the training and test data, initializes the Cellpose model, trains the model using the
    provided images and labels, and then saves the trained model to the specified training directory.

    Args:
        data_dir: The directory containing the training data, which includes the subdirectories
                   for 2D images and corresponding masks.

    Returns:
        None: This method does not return a value but prints the path where the trained model
              is saved.
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
    Tests the training of a model using the Cellpose library by executing command-line instructions.
    This method prepares the training data, removes any existing model files, and runs the training process followed by inference on the provided dataset.

    Args:
        data_dir: The directory where the training data is located.

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
    Runs a training process for a pretrained model using the Cellpose CLI.

    This method constructs the command to invoke the Cellpose training with
    specified parameters and executes it. It first removes any existing model
    directory to ensure a fresh start before running the training command.

    Args:
        data_dir: The directory containing the training data, which should include
                   a '2D/train' subdirectory with the training images.

    Returns:
        None: The method does not return any value. It may raise a ValueError
        if the command execution fails.
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
