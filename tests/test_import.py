def test_cellpose_imports_without_error():
    """
    Tests the import of the Cellpose package and its components without raising any errors.

    This method attempts to import the Cellpose library and its associated models. It ensures that the necessary classes can be initialized without issues.

    Returns:
        None: This method does not return a value.
    """
    import cellpose
    from cellpose import models, core

    model = models.CellposeModel()


def test_model_zoo_imports_without_error():
    """
    Tests the importation of models from the Cellpose library to ensure that no errors occur when loading specific models.

    This method imports the necessary components from the Cellpose library and verifies that each model,
    except those containing "neurips" or "transformer" in their names, can be instantiated without raising any exceptions.

    Returns:
        None: This method does not return a value.
    """
    from cellpose import models, denoise

    for model_name in models.MODEL_NAMES:
        if "neurips" not in model_name and "transformer" not in model_name:
            model = models.CellposeModel(model_type=model_name)


def test_gui_imports_without_error():
    """
    Test that the GUI components of the Cellpose library can be imported without errors.

    This method attempts to import the 'gui' module from the 'cellpose' package. If the import is successful, it confirms that the GUI can be accessed without encountering any import errors.

    Returns:
        None: This method does not return any value.
    """
    from cellpose import gui


def test_gpu_check():
    """
    Checks and sets the availability of GPU for Cellpose model usage.

    This method utilizes the Cellpose library's core functions to enable GPU support, allowing for accelerated processing when available.

    Returns:
        None: This method does not return any value.
    """
    #     from cellpose import models
    #     models.use_gpu()
    from cellpose import core

    core.use_gpu()


def test_model_dir():
    """
    Tests the functionality of the Cellpose model directory by evaluating a model with a random input.

    This method sets the environment variable for local Cellpose models path, initializes a Cellpose model
    with a pre-trained 'cyto3' model, and evaluates it using a random 224x224 numpy array. The method asserts
    that the output masks have the expected shape.

    Returns:
        None: This method does not return any value.
    """
    import os, pathlib
    import numpy as np

    os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = os.fspath(
        pathlib.Path.home().joinpath(".cellpose")
    )

    from cellpose import models

    model = models.CellposeModel(pretrained_model="cyto3")
    masks = model.eval(np.random.randn(224, 224))[0]
    assert masks.shape == (224, 224)
