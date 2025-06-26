"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import qtpy
from qtpy.QtWidgets import QAction
from . import io, features
from .. import models


def mainmenu(parent):
    """
    Creates and populates the main menu for the application.

        This method sets up the main menu bar by adding a file menu with various
        actions related to loading and saving images and masks. It connects
        specific keyboard shortcuts to these actions and provides options for
        autoloading masks and disabling autosave functionality.

        Args:
            parent: The parent widget that contains the menu bar.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    # load processed data
    loadImg = QAction("&Load image (*.tif, *.png, *.jpg)", parent)
    loadImg.setShortcut("Ctrl+L")
    loadImg.triggered.connect(lambda: io._load_image(parent))
    file_menu.addAction(loadImg)

    parent.autoloadMasks = QAction(
        "Autoload masks from _masks.tif file", parent, checkable=True
    )
    parent.autoloadMasks.setChecked(False)
    file_menu.addAction(parent.autoloadMasks)

    parent.disableAutosave = QAction(
        "Disable autosave _seg.npy file", parent, checkable=True
    )
    parent.disableAutosave.setChecked(False)
    file_menu.addAction(parent.disableAutosave)

    parent.loadMasks = QAction("Load &masks (*.tif, *.png, *.jpg)", parent)
    parent.loadMasks.setShortcut("Ctrl+M")
    parent.loadMasks.triggered.connect(lambda: io._load_masks(parent))
    file_menu.addAction(parent.loadMasks)
    parent.loadMasks.setEnabled(False)

    loadManual = QAction("Load &processed/labelled image (*_seg.npy)", parent)
    loadManual.setShortcut("Ctrl+P")
    loadManual.triggered.connect(lambda: io._load_seg(parent))
    file_menu.addAction(loadManual)

    parent.saveSet = QAction("&Save masks and image (as *_seg.npy)", parent)
    parent.saveSet.setShortcut("Ctrl+S")
    parent.saveSet.triggered.connect(lambda: io._save_sets(parent))
    file_menu.addAction(parent.saveSet)
    parent.saveSet.setEnabled(False)

    parent.savePNG = QAction("Save masks as P&NG/tif", parent)
    parent.savePNG.setShortcut("Ctrl+N")
    parent.savePNG.triggered.connect(lambda: io._save_png(parent))
    file_menu.addAction(parent.savePNG)
    parent.savePNG.setEnabled(False)

    parent.saveOutlines = QAction("Save &Outlines as text for imageJ", parent)
    parent.saveOutlines.setShortcut("Ctrl+O")
    parent.saveOutlines.triggered.connect(lambda: io._save_outlines(parent))
    file_menu.addAction(parent.saveOutlines)
    parent.saveOutlines.setEnabled(False)

    parent.saveROIs = QAction(
        "Save outlines as .zip archive of &ROI files for ImageJ", parent
    )
    parent.saveROIs.setShortcut("Ctrl+R")
    parent.saveROIs.triggered.connect(lambda: io._save_rois(parent))
    file_menu.addAction(parent.saveROIs)
    parent.saveROIs.setEnabled(False)

    parent.saveFlows = QAction("Save &Flows and cellprob as tif", parent)
    parent.saveFlows.setShortcut("Ctrl+F")
    parent.saveFlows.triggered.connect(lambda: io._save_flows(parent))
    file_menu.addAction(parent.saveFlows)
    parent.saveFlows.setEnabled(False)


def editmenu(parent):
    """
    Sets up the edit menu in the application's menu bar.

        This method creates an 'Edit' menu in the provided parent widget's menu bar.
        It adds several actions including undo, redo, clear masks, and remove cell
        functionalities. Each action is connected to its respective handler and
        has defined keyboard shortcuts. The actions are initially disabled.

        Args:
            parent: The parent widget that contains the menu bar for the application.

        Returns:
            None: This method does not return a value.
    """
    main_menu = parent.menuBar()
    edit_menu = main_menu.addMenu("&Edit")
    parent.undo = QAction("Undo previous mask/trace", parent)
    parent.undo.setShortcut("Ctrl+Z")
    parent.undo.triggered.connect(parent.undo_action)
    parent.undo.setEnabled(False)
    edit_menu.addAction(parent.undo)

    parent.redo = QAction("Undo remove mask", parent)
    parent.redo.setShortcut("Ctrl+Y")
    parent.redo.triggered.connect(parent.undo_remove_action)
    parent.redo.setEnabled(False)
    edit_menu.addAction(parent.redo)

    parent.ClearButton = QAction("Clear all masks", parent)
    parent.ClearButton.setShortcut("Ctrl+0")
    parent.ClearButton.triggered.connect(parent.clear_all)
    parent.ClearButton.setEnabled(False)
    edit_menu.addAction(parent.ClearButton)

    parent.remcell = QAction("Remove selected cell (Ctrl+CLICK)", parent)
    parent.remcell.setShortcut("Ctrl+Click")
    parent.remcell.triggered.connect(parent.remove_action)
    parent.remcell.setEnabled(False)
    edit_menu.addAction(parent.remcell)

    parent.mergecell = QAction("FYI: Merge cells by Alt+Click", parent)
    parent.mergecell.setEnabled(False)
    edit_menu.addAction(parent.mergecell)


def modelmenu(parent):
    """
    Creates and configures the model menu in the application's main window.

        This method initializes the model menu within the provided parent window,
        setting up actions for adding, removing, and training new custom models,
        as well as displaying training instructions.

        Args:
            parent: The parent widget that contains the menu bar. This is typically
            the main window of the application.

        Returns:
            None: This method does not return any value.
    """
    main_menu = parent.menuBar()
    io._init_model_list(parent)
    model_menu = main_menu.addMenu("&Models")
    parent.addmodel = QAction("Add custom torch model to GUI", parent)
    # parent.addmodel.setShortcut("Ctrl+A")
    parent.addmodel.triggered.connect(parent.add_model)
    parent.addmodel.setEnabled(True)
    model_menu.addAction(parent.addmodel)

    parent.removemodel = QAction("Remove selected custom model from GUI", parent)
    # parent.removemodel.setShortcut("Ctrl+R")
    parent.removemodel.triggered.connect(parent.remove_model)
    parent.removemodel.setEnabled(True)
    model_menu.addAction(parent.removemodel)

    parent.newmodel = QAction("&Train new model with image+masks in folder", parent)
    parent.newmodel.setShortcut("Ctrl+T")
    parent.newmodel.triggered.connect(parent.new_model)
    parent.newmodel.setEnabled(False)
    model_menu.addAction(parent.newmodel)

    openTrainHelp = QAction("Training instructions", parent)
    openTrainHelp.triggered.connect(parent.train_help_window)
    model_menu.addAction(openTrainHelp)


def masksmenu(parent):
    """
    Sets up the Masks menu in the application's main menu bar.

        This method creates a submenu titled 'Masks' under the main menu bar and
        adds actions for saving masks temporarily and saving labeled masks.
        The actions are connected to their respective methods in the features class
        and are initially disabled.

        Args:
            parent: The parent widget that holds the menu bar and contains
                    the features_class for handling mask-related functionalities.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    masks_menu = main_menu.addMenu("&Masks")

    parent.keepMask = QAction("Save mask temporarily", parent)
    parent.keepMask.triggered.connect(
        lambda: parent.features_class.save_temp_output(gui_self=parent)
    )
    parent.keepMask.setEnabled(False)
    masks_menu.addAction(parent.keepMask)

    parent.saveMasks = QAction("Save labeled mask", parent)
    parent.saveMasks.triggered.connect(
        lambda: parent.features_class.save_labeled_masks(gui_self=parent)
    )
    parent.saveMasks.setEnabled(False)
    masks_menu.addAction(parent.saveMasks)

    parent.features_class.main_masks_menu = masks_menu


def imagesmenu(parent):
    """
    Creates and adds an 'Images' menu to the main menu bar.

        This method initializes an 'Images' menu and adds it to the main menu bar
        of the provided parent widget. It also stores a reference to this menu in
        the parent's features_class for further manipulation.

        Args:
            parent: The parent widget to which the 'Images' menu will be added.

        Returns:
            None
    """
    main_menu = parent.menuBar()
    images_menu = main_menu.addMenu("&Images")
    parent.features_class.main_images_menu = images_menu


def helpmenu(parent):
    """
    Sets up the help menu for the application window.

    This method adds a 'Help' menu to the application’s main menu bar,
    which includes actions that open help-related windows for the user.

    Args:
        parent: The parent widget that hosts the menu bar.

    Returns:
        None
    """
    main_menu = parent.menuBar()
    help_menu = main_menu.addMenu("&Help")

    openHelp = QAction("&Help with GUI", parent)
    openHelp.setShortcut("Ctrl+H")
    openHelp.triggered.connect(parent.help_window)
    help_menu.addAction(openHelp)

    openGUI = QAction("&GUI layout", parent)
    openGUI.setShortcut("Ctrl+G")
    openGUI.triggered.connect(parent.gui_window)
    help_menu.addAction(openGUI)

    openTrainHelp = QAction("Training instructions", parent)
    openTrainHelp.triggered.connect(parent.train_help_window)
    help_menu.addAction(openTrainHelp)
