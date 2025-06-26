def test_cellpose_imports_without_error():
    """
    Test the import of the Cellpose library without raising any errors.

        This method verifies that the Cellpose library and its components can be
        successfully imported. It attempts to import the main `cellpose` module
        along with the `models` and `core` submodules, and initializes a
        `CellposeModel` instance.

        Returns:
            None: The method does not return any value. It only checks for import
            success, and any ImportError would indicate a failure in the test.
    """
    import cellpose
    from cellpose import models, core

    model = models.CellposeModel()


def test_model_zoo_imports_without_error():
    """
    Test for successful imports of model zoo components.

        This method attempts to import the necessary components from the
        cellpose library and initializes a CellposeModel for each valid
        model name, ensuring that no errors occur during the import process.

        Returns:
            None: This method does not return any values.
    """
    from cellpose import models, denoise

    for model_name in models.MODEL_NAMES:
        if "neurips" not in model_name and "transformer" not in model_name:
            model = models.CellposeModel(model_type=model_name)


def test_gui_imports_without_error():
    """
    Test if the GUI imports from the Cellpose library without raising any errors.

        This method attempts to import the GUI module from the Cellpose package to verify that the import statement executes successfully without any exceptions.

        Returns:
            None: This method does not return any value. It will raise an ImportError if the GUI module cannot be imported.
    """
    from cellpose import gui


def test_gpu_check():
    """
    Checks and enables GPU usage for the Cellpose library.

        This method verifies if the Cellpose library can utilize GPU resources
        by calling the appropriate function to enable GPU.

        Returns:
            None: This method does not return any value.
    """
    #     from cellpose import models
    #     models.use_gpu()
    from cellpose import core

    core.use_gpu()


def test_model_dir():
    """
    Tests the model directory configuration and evaluation of the Cellpose model.

        This method sets the environment variable for the local models path,
        initializes a Cellpose model with a pre-trained checkpoint, and
        evaluates the model with randomly generated data to verify the output
        shape is as expected.

        Returns:
            None: This method does not return any value. It asserts that the
            output masks shape is (224, 224).
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
