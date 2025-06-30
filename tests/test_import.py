def test_cellpose_imports_without_error():
    """


    Tests the import of the Cellpose library and its components without raising any errors.



    This method imports the main Cellpose package as well as its models and core modules,

    and attempts to instantiate a CellposeModel to ensure that the imports are functioning

    correctly.



    Returns:

        None: This method does not return any value.

    """
    import cellpose
    from cellpose import models, core

    model = models.CellposeModel()


def test_model_zoo_imports_without_error():
    """
    Tests the import of model zoo components without raising any errors.

    This method imports the models and denoise functions from the cellpose package,
    and iterates through the available model names, creating an instance of
    CellposeModel for each model that does not have 'neurips' or 'transformer'
    in its name.

    Returns:
        None: This method does not return any value.
    """
    from cellpose import models, denoise

    for model_name in models.MODEL_NAMES:
        if "neurips" not in model_name and "transformer" not in model_name:
            model = models.CellposeModel(model_type=model_name)


def test_gui_imports_without_error():
    """
    Tests the GUI imports from the Cellpose package to ensure they do not raise any errors.

    Parameters:
    None

    Returns:
    None
    """
    from cellpose import gui


def test_gpu_check():
    """
    Checks and enables GPU usage for the Cellpose library.

    This method configures the Cellpose library to utilize the GPU if available, which can improve performance during image processing tasks.

    Returns:
        None: This method does not return any value.
    """
    #     from cellpose import models
    #     models.use_gpu()
    from cellpose import core

    core.use_gpu()


def test_model_dir():
    """
    Tests the behavior of the Cellpose model by ensuring that it can generate masks for a random input.

    This method sets the local models path for Cellpose and then initializes a Cellpose model with pre-trained parameters.
    It generates a mask from a random 224x224 input and asserts that the output mask has the expected shape.

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
