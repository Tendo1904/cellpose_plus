"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, pathlib, warnings, datetime, time, copy, math

from qtpy import QtGui, QtCore
from superqt import QRangeSlider, QCollapsible
from qtpy.QtWidgets import (
    QScrollArea,
    QMainWindow,
    QAction,
    QMenu,
    QApplication,
    QWidget,
    QScrollBar,
    QComboBox,
    QGridLayout,
    QPushButton,
    QFrame,
    QCheckBox,
    QLabel,
    QProgressBar,
    QLineEdit,
    QMessageBox,
    QGroupBox,
)
import pyqtgraph as pg

import pandas as pd
import numpy as np
from scipy.stats import mode
import cv2

from . import guiparts, menus, io, symmetry, features
from .. import models, core, dynamics, version, denoise, train
from ..utils import download_url_to_file, masks_to_outlines, diameters, download_font
from ..io import get_image_files, imsave, imread
from ..transforms import resize_image, normalize99, normalize99_tile, smooth_sharpen_img
from ..models import normalize_default
from ..plot import disk

from scipy.ndimage import find_objects
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import ndimage
import diplib as dip
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False

try:
    from google.cloud import storage

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "key/cellpose-data-writer.json"
    )
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

Horizontal = QtCore.Qt.Orientation.Horizontal


class Slider(QRangeSlider):
    """
    A class that represents a slider widget, allowing users to select a value
    within a defined range.

    This class provides functionality to notify its parent widget when the
    slider's value changes, enabling dynamic interaction within the application.

    Attributes:
        parent: The parent widget that this slider is associated with.
        name: The name of the slider for identification or display purposes.
        color: The color associated with the slider, although it is not used
               in this initialization.

    Methods:
        __init__ : Initializes a new instance of the class.
        levelChanged: Notify the parent about a level change.

    The __init__ method sets up the slider by connecting its value change event
    to a callback, applying a stylesheet for visual customization, and setting
    the initial state of the slider to disabled. The levelChanged method informs
    the parent object that the level of the current object has changed, triggering
    any corresponding actions or updates within the parent.
    """

    def __init__(self, parent, name, color):
        """
        Initializes a new instance of the class.

            This method sets up the slider by connecting its value change event to a callback,
            applying a stylesheet for visual customization, and setting the initial state of the
            slider to disabled.

            Args:
                parent: The parent widget that this slider is associated with.
                name: The name of the slider for identification or display purposes.
                color: The color associated with the slider, although it is not used in this initialization.

            Returns:
                None
        """
        super().__init__(Horizontal)
        self.setEnabled(False)
        self.valueChanged.connect(lambda: self.levelChanged(parent))
        self.name = name

        self.setStyleSheet(
            """ QSlider{
                             background-color: transparent;
                             }
        """
        )
        self.show()

    def levelChanged(self, parent):
        """
        Notify the parent about a level change.

            This method informs the parent object that the level of the current
            object has changed, triggering any corresponding actions or updates
            within the parent.

            Args:
                parent: The parent object that will receive a notification
                    of the level change.

            Returns:
                None
        """
        parent.level_change(self.name)


class QHLine(QFrame):
    """
    A class that represents a horizontal line frame.

        This class is responsible for creating a visual representation of a horizontal
        line in a graphical interface. It inherits from its parent class and customizes
        the appearance of the frame to be a horizontal line.

        Methods:
            __init__

        Attributes:
            None

        The __init__ method initializes the QHLine object by setting the frame shape
        to a horizontal line and configuring its line width. This ensures that the
        line is displayed correctly in the user interface.
    """

    def __init__(self):
        """
        Initializes a horizontal line frame.

            This method initializes the QHLine object by calling the parent class's
            constructor and setting the frame shape to a horizontal line. It also
            configures the line width for the frame.

            Parameters:
                None

            Returns:
                None
        """
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        # self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(8)


def make_bwr():
    """
    Create a blue-white-red colormap.

        This method generates a colormap that smoothly transitions from blue
        to white to red. The colormap is composed of a gradient of colors
        using a linear interpolation of RGB values. It is useful for visualizing
        data where a diverging color scheme is desirable.

        Returns:
            A ColorMap object representing the blue-white-red colormap.
    """
    # make a bwr colormap
    b = np.append(255 * np.ones(128), np.linspace(0, 255, 128)[::-1])[:, np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255 * np.ones(128))[:, np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[
        :, np.newaxis
    ]
    color = np.concatenate((r, g, b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return bwr


def make_spectral():
    """
    Create a spectral colormap.

        This method generates a spectral colormap using predefined red, green,
        and blue channel values. The colors are combined to create an RGB
        representation which is then formatted into a colormap object.

        Returns:
            A colormap object that can be used for visualizations.
    """
    # make spectral colormap
    r = np.array(
        [
            0,
            4,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            68,
            72,
            76,
            80,
            84,
            88,
            92,
            96,
            100,
            104,
            108,
            112,
            116,
            120,
            124,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            120,
            112,
            104,
            96,
            88,
            80,
            72,
            64,
            56,
            48,
            40,
            32,
            24,
            16,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            7,
            11,
            15,
            19,
            23,
            27,
            31,
            35,
            39,
            43,
            47,
            51,
            55,
            59,
            63,
            67,
            71,
            75,
            79,
            83,
            87,
            91,
            95,
            99,
            103,
            107,
            111,
            115,
            119,
            123,
            127,
            131,
            135,
            139,
            143,
            147,
            151,
            155,
            159,
            163,
            167,
            171,
            175,
            179,
            183,
            187,
            191,
            195,
            199,
            203,
            207,
            211,
            215,
            219,
            223,
            227,
            231,
            235,
            239,
            243,
            247,
            251,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
        ]
    )
    g = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            9,
            9,
            8,
            8,
            7,
            7,
            6,
            6,
            5,
            5,
            5,
            4,
            4,
            3,
            3,
            2,
            2,
            1,
            1,
            0,
            0,
            0,
            7,
            15,
            23,
            31,
            39,
            47,
            55,
            63,
            71,
            79,
            87,
            95,
            103,
            111,
            119,
            127,
            135,
            143,
            151,
            159,
            167,
            175,
            183,
            191,
            199,
            207,
            215,
            223,
            231,
            239,
            247,
            255,
            247,
            239,
            231,
            223,
            215,
            207,
            199,
            191,
            183,
            175,
            167,
            159,
            151,
            143,
            135,
            128,
            129,
            131,
            132,
            134,
            135,
            137,
            139,
            140,
            142,
            143,
            145,
            147,
            148,
            150,
            151,
            153,
            154,
            156,
            158,
            159,
            161,
            162,
            164,
            166,
            167,
            169,
            170,
            172,
            174,
            175,
            177,
            178,
            180,
            181,
            183,
            185,
            186,
            188,
            189,
            191,
            193,
            194,
            196,
            197,
            199,
            201,
            202,
            204,
            205,
            207,
            208,
            210,
            212,
            213,
            215,
            216,
            218,
            220,
            221,
            223,
            224,
            226,
            228,
            229,
            231,
            232,
            234,
            235,
            237,
            239,
            240,
            242,
            243,
            245,
            247,
            248,
            250,
            251,
            253,
            255,
            251,
            247,
            243,
            239,
            235,
            231,
            227,
            223,
            219,
            215,
            211,
            207,
            203,
            199,
            195,
            191,
            187,
            183,
            179,
            175,
            171,
            167,
            163,
            159,
            155,
            151,
            147,
            143,
            139,
            135,
            131,
            127,
            123,
            119,
            115,
            111,
            107,
            103,
            99,
            95,
            91,
            87,
            83,
            79,
            75,
            71,
            67,
            63,
            59,
            55,
            51,
            47,
            43,
            39,
            35,
            31,
            27,
            23,
            19,
            15,
            11,
            7,
            3,
            0,
            8,
            16,
            24,
            32,
            41,
            49,
            57,
            65,
            74,
            82,
            90,
            98,
            106,
            115,
            123,
            131,
            139,
            148,
            156,
            164,
            172,
            180,
            189,
            197,
            205,
            213,
            222,
            230,
            238,
            246,
            254,
        ]
    )
    b = np.array(
        [
            0,
            7,
            15,
            23,
            31,
            39,
            47,
            55,
            63,
            71,
            79,
            87,
            95,
            103,
            111,
            119,
            127,
            135,
            143,
            151,
            159,
            167,
            175,
            183,
            191,
            199,
            207,
            215,
            223,
            231,
            239,
            247,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            251,
            247,
            243,
            239,
            235,
            231,
            227,
            223,
            219,
            215,
            211,
            207,
            203,
            199,
            195,
            191,
            187,
            183,
            179,
            175,
            171,
            167,
            163,
            159,
            155,
            151,
            147,
            143,
            139,
            135,
            131,
            128,
            126,
            124,
            122,
            120,
            118,
            116,
            114,
            112,
            110,
            108,
            106,
            104,
            102,
            100,
            98,
            96,
            94,
            92,
            90,
            88,
            86,
            84,
            82,
            80,
            78,
            76,
            74,
            72,
            70,
            68,
            66,
            64,
            62,
            60,
            58,
            56,
            54,
            52,
            50,
            48,
            46,
            44,
            42,
            40,
            38,
            36,
            34,
            32,
            30,
            28,
            26,
            24,
            22,
            20,
            18,
            16,
            14,
            12,
            10,
            8,
            6,
            4,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            8,
            16,
            24,
            32,
            41,
            49,
            57,
            65,
            74,
            82,
            90,
            98,
            106,
            115,
            123,
            131,
            139,
            148,
            156,
            164,
            172,
            180,
            189,
            197,
            205,
            213,
            222,
            230,
            238,
            246,
            254,
        ]
    )
    color = (np.vstack((r, g, b)).T).astype(np.uint8)
    spectral = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return spectral


def make_cmap(cm=0):
    """
    Create a single channel colormap.

        This method generates a colormap where the specified channel (red, green, or blue) is
        filled with a gradient from 0 to 255, while the other channels are set to zero.
        The resulting colormap can be used for visualizing data with a single color channel.

        Args:
            cm: The index of the color channel to fill (0 for red, 1 for green, and 2 for blue).

        Returns:
            A ColorMap object representing the single channel colormap.
    """
    # make a single channel colormap
    r = np.arange(0, 256)
    color = np.zeros((256, 3))
    color[:, cm] = r
    color = color.astype(np.uint8)
    cmap = pg.ColorMap(pos=np.linspace(0.0, 255, 256), color=color)
    return cmap


def run(image=None):
    """
    Runs the main application for cell segmentation.

        This method initializes the Qt application, sets up logging,
        downloads necessary icon and image files for the GUI, and starts
        the main application window for cell segmentation using the Cellpose
        framework.

        Args:
            image: Optional; the image to be processed by the application.
                   If provided, the application will use this image for
                   analysis.

        Returns:
            int: The exit status of the application upon closure.
    """
    from ..io import logger_setup

    logger, log_file = logger_setup()
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
    guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")

    primary_icon_path = pathlib.Path.home().joinpath(".cellpose", "primary.png")
    primary_icon_url = "https://github.com/ITMO-MMRM-lab/cellpose/blob/main/cellpose/resources/primary.png?raw=true"

    secondary_icon_path = pathlib.Path.home().joinpath(".cellpose", "secondary.png")
    secondary_icon_url = "https://github.com/ITMO-MMRM-lab/cellpose/blob/main/cellpose/resources/secondary.png?raw=true"

    if not icon_path.is_file():
        cp_dir = pathlib.Path.home().joinpath(".cellpose")
        cp_dir.mkdir(exist_ok=True)
        print("downloading logo")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_transparent.png",
            icon_path,
            progress=True,
        )
    if not guip_path.is_file():
        print("downloading help window image")
        download_url_to_file(
            "https://www.cellpose.org/static/images/cellpose_gui.png",
            guip_path,
            progress=True,
        )
    if not primary_icon_path.is_file():
        print("downloading primary mask image")
        download_url_to_file(primary_icon_url, primary_icon_path, progress=True)
    if not secondary_icon_path.is_file():
        print("downloading secondary mask image")
        download_url_to_file(secondary_icon_url, secondary_icon_path, progress=True)

    download_font()

    icon_path = str(icon_path.resolve())
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setStyle("Fusion")
    app.setPalette(guiparts.DarkPalette())
    # app.setStyleSheet("QLineEdit { color: yellow }")

    # models.download_model_weights() # does not exist
    MainW(image=image, logger=logger)
    ret = app.exec_()
    sys.exit(ret)


class MainW(QMainWindow):
    """
    MainW is a class that represents the main application window for the Cellpose application,
    featuring a graphical user interface (GUI) designed for image processing and segmentation.
    It manages the configuration and functionality of various UI components, allowing users
    to interact with images, models, and settings efficiently.

    Methods:
        - __init__
        - help_window
        - train_help_window
        - gui_window
        - make_buttons
        - update_px_to_mm
        - level_change
        - keyPressEvent
        - autosave_on
        - check_gpu
        - get_channels
        - model_choose
        - calibrate_size
        - toggle_scale
        - enable_buttons
        - disable_buttons_removeROIs
        - toggle_mask_ops
        - toggle_saving
        - toggle_removals
        - remove_action
        - undo_action
        - undo_remove_action
        - get_files
        - get_prev_image
        - get_next_image
        - dragEnterEvent
        - dropEvent
        - toggle_masks
        - make_viewbox
        - reset
        - clear_all
        - select_cell
        - select_cell_multi
        - unselect_cell
        - unselect_cell_multi
        - remove_cell
        - remove_single_cell
        - remove_region_cells
        - delete_multiple_cells
        - done_remove_multiple_cells
        - merge_cells
        - undo_remove_cell
        - remove_stroke
        - plot_clicked
        - cancel_remove_multiple
        - clear_multi_selected_cells
        - add_roi
        - remove_roi
        - roi_changed
        - mouse_moved
        - color_choose
        - update_plot
        - update_layer
        - update_roi_count
        - add_set
        - add_mask
        - draw_mask
        - compute_scale
        - update_scale
        - redraw_masks
        - draw_masks
        - draw_layer
        - set_restore_button
        - set_normalize_params
        - check_percentile_params
        - check_filter_params
        - get_normalize_params
        - compute_saturation
        - chanchoose
        - get_model_path
        - initialize_model
        - add_model
        - remove_model
        - new_model
        - train_model
        - compute_restore
        - get_thresholds
        - compute_cprob
        - compute_denoise_model
        - compute_segmentation

    Attributes:
        - px_to_mm
        - model_strings
        - StyleButtons
        - DenoiseButtons
        - load_3D
        - sliders
        - nchan
        - filename
        - masksOn
        - outlinesOn
        - calcSize
        - calcRound
        - calcRatio
        - calcVoronoi
        - ncells
        - brush_size
        - diameter
        - pr
        - radii_padding
        - radii

    The methods in this class encompass a variety of functionalities including
    initializing and managing the GUI, handling user interactions, performing image
    processing tasks, managing model training, and updating various attributes
    based on user input and application state. The attributes primarily store
    configuration settings, control elements of the GUI, and hold important data
    for image processing.
    """

    def __init__(self, image=None, logger=None):
        """
        Initializes the MainW object.

            This method sets up the main window for the cellpose application,
            configuring its appearance, layout, and functionality. It initializes
            various UI components, style settings, and prepares the main
            functionalities for image processing.

            Args:
                image: Optional; a string representing the file path of an image to
                    load at initialization. If not provided, no image will be loaded.
                logger: Optional; a logging instance for capturing application
                    events and messages. If not provided, logging may be disabled.

            Returns:
                None
        """
        super(MainW, self).__init__()

        self.logger = logger
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 1200, 1000)
        self.setWindowTitle(f"cellpose v{version}")
        self.cp_path = os.path.dirname(os.path.realpath(__file__))
        app_icon = QtGui.QIcon()
        icon_path = pathlib.Path.home().joinpath(".cellpose", "logo.png")
        icon_path = str(icon_path.resolve())
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(64, 64))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        # rgb(150,255,150)
        self.setStyleSheet(guiparts.stylesheet())

        self.main_masks_menu = None  # Pointer to masks menu
        self.main_images_menu = None  # Pointer to images menu
        self.temp_masks = []
        self.px_to_mm = 0.0
        self.selected_model = None

        self.features_class = features.FeatureExtraction()

        menus.mainmenu(self)
        menus.editmenu(self)
        menus.modelmenu(self)
        menus.masksmenu(self)
        menus.imagesmenu(self)
        menus.helpmenu(self)

        self.stylePressed = """QPushButton {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.styleUnpressed = """QPushButton {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }"""
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QWidget(self)
        self.lmain = QGridLayout()
        self.cwidget.setLayout(self.lmain)
        self.setCentralWidget(self.cwidget)
        self.lmain.setVerticalSpacing(0)
        self.lmain.setContentsMargins(0, 0, 0, 10)

        self.imask = 0
        self.scrollarea = QScrollArea()
        self.scrollarea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollarea.setStyleSheet("""QScrollArea { border: none }""")
        self.scrollarea.setWidgetResizable(True)
        self.swidget = QWidget(self)
        self.scrollarea.setWidget(self.swidget)
        self.l0 = QGridLayout()
        self.swidget.setLayout(self.l0)
        b = self.make_buttons()
        self.lmain.addWidget(self.scrollarea, 0, 0, 39, 9)

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()

        self.lmain.addWidget(self.win, 0, 9, 40, 30)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        self.lmain.setColumnStretch(10, 1)
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)
        self.cmap = []
        # spectral colormap
        self.cmap.append(
            make_spectral().getLookupTable(start=0.0, stop=255.0, alpha=False)
        )
        # single channel colormaps
        for i in range(3):
            self.cmap.append(
                make_cmap(i).getLookupTable(start=0.0, stop=255.0, alpha=False)
            )

        if MATPLOTLIB:
            self.colormap = (
                plt.get_cmap("gist_ncar")(np.linspace(0.0, 0.9, 1000000)) * 255
            ).astype(np.uint8)
            np.random.seed(42)  # make colors stable
            self.colormap = self.colormap[np.random.permutation(1000000)]
        else:
            np.random.seed(42)  # make colors stable
            self.colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(
                np.uint8
            )
        self.NZ = 1
        self.restore = None
        self.ratio = 1.0
        self.reset()

        # if called with image, load it
        if image is not None:
            self.filename = image
            io._load_image(self, self.filename)

        # training settings
        d = datetime.datetime.now()
        self.training_params = {
            "model_index": 0,
            "learning_rate": 0.1,
            "weight_decay": 0.0001,
            "n_epochs": 100,
            "SGD": True,
            "model_name": "CP" + d.strftime("_%Y%m%d_%H%M%S"),
        }

        self.load_3D = False
        self.stitch_threshold = 0.0
        self.flow3D_smooth = 0.0
        self.anisotropy = 1.0
        self.min_size = 15
        self.resample = True

        self.setAcceptDrops(True)
        self.win.show()
        self.show()

    def help_window(self):
        """
        Display the help window.

            This method creates an instance of the HelpWindow from the guiparts
            module and displays it to the user.

            Returns:
                None: This method does not return any value.
        """
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        """
        Display the Train Help Window.

            This method creates an instance of the TrainHelpWindow and displays it to the user.

            Parameters:
                None

            Returns:
                None
        """
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        """
        Launches the graphical user interface window.

            This method initializes an instance of the ExampleGUI class and displays
            it to the user. It serves as the entry point for the graphical interface
            of the application.

            Parameters:
                None

            Returns:
                None
        """
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        """
        Create and layout various GUI buttons and controls for the application.

            This method initializes buttons, checkboxes, dropdown menus, and sliders for controlling
            various aspects of the GUI, including image viewing options, drawing parameters, segmentation,
            and metrics calculations. It organizes these controls into group boxes with appropriate labels
            and layouts.

            The method sets up various GUI elements, including:
            - Buttons for region selection and deletion of multiple ROIs
            - Dropdowns for selecting color channels and views
            - Sliders for adjusting saturation levels
            - Checkboxes for enabling or disabling masks and outlines
            - Controls for segmentation settings, including thresholds and normalization settings

            The user can interact with these controls to modify the behavior of the application,
            which is primarily focused on image analysis and segmentation tasks.

            Returns:
                int: The updated row index after adding all controls to the layout.
        """
        self.boldfont = QtGui.QFont("Arial", 11, QtGui.QFont.Bold)
        self.boldmedfont = QtGui.QFont("Arial", 9, QtGui.QFont.Bold)
        self.medfont = QtGui.QFont("Arial", 9)
        self.smallfont = QtGui.QFont("Arial", 8)

        b = 0
        self.satBox = QGroupBox("Views")
        self.satBox.setFont(self.boldfont)
        self.satBoxG = QGridLayout()
        self.satBox.setLayout(self.satBoxG)
        self.l0.addWidget(self.satBox, b, 0, 1, 9)

        b0 = 0
        self.view = 0  # 0=image, 1=flowsXY, 2=flowsZ, 3=cellprob
        self.color = 0  # 0=RGB, 1=gray, 2=R, 3=G, 4=B
        self.RGBDropDown = QComboBox()
        self.RGBDropDown.addItems(
            ["RGB", "red=R", "green=G", "blue=B", "gray", "spectral"]
        )
        self.RGBDropDown.setFont(self.medfont)
        self.RGBDropDown.currentIndexChanged.connect(self.color_choose)
        self.satBoxG.addWidget(self.RGBDropDown, b0, 0, 1, 3)

        label = QLabel("<p>[&uarr; / &darr; or W/S]</p>")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 3)
        label = QLabel("[R / G / B \n toggles color ]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.ViewDropDown = QComboBox()
        self.ViewDropDown.addItems(["image", "gradXY", "cellprob", "restored"])
        self.ViewDropDown.setFont(self.medfont)
        self.ViewDropDown.model().item(3).setEnabled(False)
        self.ViewDropDown.currentIndexChanged.connect(self.update_plot)
        self.satBoxG.addWidget(self.ViewDropDown, b0, 0, 2, 3)

        label = QLabel("[pageup / pagedown]")
        label.setFont(self.smallfont)
        self.satBoxG.addWidget(label, b0, 3, 1, 5)

        b0 += 2
        label = QLabel("")
        label.setToolTip(
            "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
        )
        self.satBoxG.addWidget(label, b0, 0, 1, 5)

        self.autobtn = QCheckBox("auto-adjust saturation")
        self.autobtn.setToolTip("sets scale-bars as normalized for segmentation")
        self.autobtn.setFont(self.medfont)
        self.autobtn.setChecked(True)
        self.satBoxG.addWidget(self.autobtn, b0, 1, 1, 8)

        b0 += 1
        self.sliders = []
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100]]
        colornames = ["red", "Chartreuse", "DodgerBlue"]
        names = ["red", "green", "blue"]
        for r in range(3):
            b0 += 1
            if r == 0:
                label = QLabel('<font color="gray">gray/</font><br>red')
            else:
                label = QLabel(names[r] + ":")
            label.setStyleSheet(f"color: {colornames[r]}")
            label.setFont(self.boldmedfont)
            self.satBoxG.addWidget(label, b0, 0, 1, 2)
            self.sliders.append(Slider(self, names[r], colors[r]))
            self.sliders[-1].setMinimum(-0.1)
            self.sliders[-1].setMaximum(255.1)
            self.sliders[-1].setValue([0, 255])
            self.sliders[-1].setToolTip(
                "NOTE: manually changing the saturation bars does not affect normalization in segmentation"
            )
            # self.sliders[-1].setTickPosition(QSlider.TicksRight)
            self.satBoxG.addWidget(self.sliders[-1], b0, 2, 1, 7)

        b += 1
        self.drawBox = QGroupBox("Drawing")
        self.drawBox.setFont(self.boldfont)
        self.drawBoxG = QGridLayout()
        self.drawBox.setLayout(self.drawBoxG)
        self.l0.addWidget(self.drawBox, b, 0, 1, 9)
        self.autosave = True

        b0 = 0
        self.brush_size = 3
        self.BrushChoose = QComboBox()
        self.BrushChoose.addItems(["1", "3", "5", "7", "9"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.BrushChoose.setFixedWidth(40)
        self.BrushChoose.setFont(self.medfont)
        self.drawBoxG.addWidget(self.BrushChoose, b0, 3, 1, 2)
        label = QLabel("brush size:")
        label.setFont(self.medfont)
        self.drawBoxG.addWidget(label, b0, 0, 1, 3)

        b0 += 1
        # turn off masks
        self.layer_off = False
        self.masksOn = True
        self.MCheckBox = QCheckBox("MASKS ON [X]")
        self.MCheckBox.setFont(self.medfont)
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.toggle_masks)
        self.drawBoxG.addWidget(self.MCheckBox, b0, 0, 1, 5)

        b0 += 1
        # turn off outlines
        self.outlinesOn = False  # turn off by default
        self.OCheckBox = QCheckBox("outlines on [Z]")
        self.OCheckBox.setFont(self.medfont)
        self.drawBoxG.addWidget(self.OCheckBox, b0, 0, 1, 5)
        self.OCheckBox.setChecked(False)
        self.OCheckBox.toggled.connect(self.toggle_masks)

        b0 += 1
        self.SCheckBox = QCheckBox("single stroke")
        self.SCheckBox.setFont(self.medfont)
        self.SCheckBox.setChecked(True)
        self.SCheckBox.toggled.connect(self.autosave_on)
        self.SCheckBox.setEnabled(True)
        self.drawBoxG.addWidget(self.SCheckBox, b0, 0, 1, 5)

        # buttons for deleting multiple cells
        self.deleteBox = QGroupBox("delete multiple ROIs")
        self.deleteBox.setStyleSheet("color: rgb(200, 200, 200)")
        self.deleteBox.setFont(self.medfont)
        self.deleteBoxG = QGridLayout()
        self.deleteBox.setLayout(self.deleteBoxG)
        self.drawBoxG.addWidget(self.deleteBox, 0, 5, 4, 4)
        self.MakeDeletionRegionButton = QPushButton("region-select")
        self.MakeDeletionRegionButton.clicked.connect(self.remove_region_cells)
        self.deleteBoxG.addWidget(self.MakeDeletionRegionButton, 0, 0, 1, 4)
        self.MakeDeletionRegionButton.setFont(self.smallfont)
        self.MakeDeletionRegionButton.setFixedWidth(70)
        self.DeleteMultipleROIButton = QPushButton("click-select")
        self.DeleteMultipleROIButton.clicked.connect(self.delete_multiple_cells)
        self.deleteBoxG.addWidget(self.DeleteMultipleROIButton, 1, 0, 1, 4)
        self.DeleteMultipleROIButton.setFont(self.smallfont)
        self.DeleteMultipleROIButton.setFixedWidth(70)
        self.DoneDeleteMultipleROIButton = QPushButton("done")
        self.DoneDeleteMultipleROIButton.clicked.connect(
            self.done_remove_multiple_cells
        )
        self.deleteBoxG.addWidget(self.DoneDeleteMultipleROIButton, 2, 0, 1, 2)
        self.DoneDeleteMultipleROIButton.setFont(self.smallfont)
        self.DoneDeleteMultipleROIButton.setFixedWidth(35)
        self.CancelDeleteMultipleROIButton = QPushButton("cancel")
        self.CancelDeleteMultipleROIButton.clicked.connect(self.cancel_remove_multiple)
        self.deleteBoxG.addWidget(self.CancelDeleteMultipleROIButton, 2, 2, 1, 2)
        self.CancelDeleteMultipleROIButton.setFont(self.smallfont)
        self.CancelDeleteMultipleROIButton.setFixedWidth(35)

        b += 1
        b0 = 0
        self.segBox = QGroupBox("Segmentation")
        self.segBoxG = QGridLayout()
        self.segBox.setLayout(self.segBoxG)
        self.l0.addWidget(self.segBox, b, 0, 1, 9)
        self.segBox.setFont(self.boldfont)

        self.diameter = 30
        label = QLabel("diameter (pixels):")
        label.setFont(self.medfont)
        label.setToolTip(
            "you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)"
        )
        self.segBoxG.addWidget(label, b0, 0, 1, 4)
        self.Diameter = QLineEdit()
        self.Diameter.setToolTip(
            'you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the "cyto3" model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)'
        )
        self.Diameter.setText(str(self.diameter))
        self.Diameter.setFont(self.medfont)
        self.Diameter.returnPressed.connect(self.update_scale)
        self.Diameter.setFixedWidth(50)
        self.segBoxG.addWidget(self.Diameter, b0, 4, 1, 2)

        # compute diameter
        self.SizeButton = QPushButton("calibrate")
        self.SizeButton.setFont(self.medfont)
        self.SizeButton.clicked.connect(self.calibrate_size)
        self.segBoxG.addWidget(self.SizeButton, b0, 6, 1, 3)
        # self.SizeButton.setFixedWidth(65)
        self.SizeButton.setEnabled(False)
        self.SizeButton.setToolTip(
            "you can manually enter the approximate diameter for your cells, \nor press “calibrate” to let the cyto3 model estimate it. \nThe size is represented by a disk at the bottom of the view window \n(can turn this disk off by unchecking “scale disk on”)"
        )

        b0 += 1
        label = QLabel("Length in μm:")
        label.setToolTip("Micrometers(μm) per pixel, *.tif file")
        label.setFont(self.medfont)
        self.segBoxG.addWidget(label, b0, 0, 1, 4)

        self.pixTomicro = QLineEdit()
        self.pixTomicro.setText("0.0")
        self.pixTomicro.editingFinished.connect(self.update_px_to_mm)
        self.pixTomicro.setFixedWidth(70)
        self.segBoxG.addWidget(self.pixTomicro, b0, 4, 1, 2)

        b0 += 1
        # choose channel
        self.ChannelChoose = [QComboBox(), QComboBox()]
        self.ChannelChoose[0].addItems(["0: gray", "1: red", "2: green", "3: blue"])
        self.ChannelChoose[1].addItems(["0: none", "1: red", "2: green", "3: blue"])
        cstr = ["chan to segment:", "chan2 (optional): "]
        for i in range(2):
            self.ChannelChoose[i].setFont(self.medfont)
            label = QLabel(cstr[i])
            label.setFont(self.medfont)
            if i == 0:
                label.setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
                self.ChannelChoose[i].setToolTip(
                    "this is the channel in which the cytoplasm or nuclei exist that you want to segment"
                )
            else:
                label.setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
                self.ChannelChoose[i].setToolTip(
                    "if <em>cytoplasm</em> model is chosen, and you also have a nuclear channel, then choose the nuclear channel for this option"
                )
            self.segBoxG.addWidget(label, b0 + i, 0, 1, 4)
            self.segBoxG.addWidget(self.ChannelChoose[i], b0 + i, 4, 1, 5)

        b0 += 2

        # use GPU
        self.useGPU = QCheckBox("use GPU")
        self.useGPU.setToolTip(
            "if you have specially installed the <i>cuda</i> version of torch, then you can activate this"
        )
        self.useGPU.setFont(self.medfont)
        self.check_gpu()
        self.segBoxG.addWidget(self.useGPU, b0, 0, 1, 3)

        # compute segmentation with general models
        self.net_text = ["run cyto3"]
        nett = ["cellpose super-generalist model"]

        # label = QLabel("Run:")
        # label.setFont(self.boldfont)
        # label.setFont(self.medfont)
        # self.segBoxG.addWidget(label, b0, 0, 1, 2)
        self.StyleButtons = []
        jj = 4
        for j in range(len(self.net_text)):
            self.StyleButtons.append(
                guiparts.ModelButton(self, self.net_text[j], self.net_text[j])
            )
            w = 5
            self.segBoxG.addWidget(self.StyleButtons[-1], b0, jj, 1, w)
            jj += w
            # self.StyleButtons[-1].setFixedWidth(140)
            self.StyleButtons[-1].setToolTip(nett[j])

        b0 += 1
        self.roi_count = QLabel("0 ROIs")
        self.roi_count.setFont(self.boldfont)
        self.roi_count.setAlignment(QtCore.Qt.AlignLeft)
        self.segBoxG.addWidget(self.roi_count, b0, 0, 1, 4)

        self.progress = QProgressBar(self)
        self.segBoxG.addWidget(self.progress, b0, 4, 1, 5)

        b0 += 1
        self.segaBox = QCollapsible("additional settings")
        self.segaBox.setFont(self.medfont)
        self.segaBox._toggle_btn.setFont(self.medfont)
        self.segaBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.segaBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        # _content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.segaBox.setContent(_content)
        self.segBoxG.addWidget(self.segaBox, b0, 0, 1, 9)

        b0 = 0
        # post-hoc paramater tuning
        label = QLabel("flow\nthreshold:")
        label.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 2)
        self.flow_threshold = QLineEdit()
        self.flow_threshold.setText("0.4")
        self.flow_threshold.returnPressed.connect(self.compute_cprob)
        self.flow_threshold.setFixedWidth(40)
        self.flow_threshold.setFont(self.medfont)
        self.segaBoxG.addWidget(self.flow_threshold, b0, 2, 1, 2)
        self.flow_threshold.setToolTip(
            "threshold on flow error to accept a mask (set higher to get more cells, e.g. in range from (0.1, 3.0), OR set to 0.0 to turn off so no cells discarded);\n press enter to recompute if model already run"
        )

        label = QLabel("cellprob\nthreshold:")
        label.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 4, 1, 2)
        self.cellprob_threshold = QLineEdit()
        self.cellprob_threshold.setText("0.0")
        self.cellprob_threshold.returnPressed.connect(self.compute_cprob)
        self.cellprob_threshold.setFixedWidth(40)
        self.cellprob_threshold.setFont(self.medfont)
        self.cellprob_threshold.setToolTip(
            "threshold on cellprob output to seed cell masks (set lower to include more pixels or higher to include fewer, e.g. in range from (-6, 6)); \n press enter to recompute if model already run"
        )
        self.segaBoxG.addWidget(self.cellprob_threshold, b0, 6, 1, 2)

        b0 += 1
        label = QLabel("norm percentiles:")
        label.setToolTip(
            "sets normalization percentiles for segmentation and denoising\n(pixels at lower percentile set to 0.0 and at upper set to 1.0 for network)"
        )
        label.setFont(self.medfont)
        self.segaBoxG.addWidget(label, b0, 0, 1, 8)

        b0 += 1
        self.norm_vals = [1.0, 99.0]
        self.norm_edits = []
        labels = ["lower", "upper"]
        tooltips = [
            "pixels at this percentile set to 0 (default 1.0)",
            "pixels at this percentile set to 1  (default 99.0)",
        ]
        for p in range(2):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.segaBoxG.addWidget(label, b0, 4 * (p % 2), 1, 2)
            self.norm_edits.append(QLineEdit())
            self.norm_edits[p].setText(str(self.norm_vals[p]))
            self.norm_edits[p].setFixedWidth(40)
            self.norm_edits[p].setFont(self.medfont)
            self.segaBoxG.addWidget(self.norm_edits[p], b0, 4 * (p % 2) + 2, 1, 2)
            self.norm_edits[p].setToolTip(tooltips[p])

        b0 += 1
        label = QLabel("niter dynamics:")
        label.setFont(self.medfont)
        label.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(label, b0, 0, 1, 4)
        self.niter = QLineEdit()
        self.niter.setText("0")
        self.niter.setFixedWidth(40)
        self.niter.setFont(self.medfont)
        self.niter.setToolTip(
            "number of iterations for dynamics (0 uses default based on diameter); use 2000 for bacteria"
        )
        self.segaBoxG.addWidget(self.niter, b0, 4, 1, 2)

        b += 1
        b0 = 0
        self.modelBox = QGroupBox("Other models")
        self.modelBoxG = QGridLayout()
        self.modelBox.setLayout(self.modelBoxG)
        self.l0.addWidget(self.modelBox, b, 0, 1, 9)
        self.modelBox.setFont(self.boldfont)
        # choose models
        self.ModelChooseC = QComboBox()
        self.ModelChooseC.setFont(self.medfont)
        current_index = 0
        self.ModelChooseC.addItems(["custom models"])
        if len(self.model_strings) > 0:
            self.ModelChooseC.addItems(self.model_strings)
        self.ModelChooseC.setFixedWidth(175)
        self.ModelChooseC.setCurrentIndex(current_index)
        tipstr = 'add or train your own models in the "Models" file menu and choose model here'
        self.ModelChooseC.setToolTip(tipstr)
        self.ModelChooseC.activated.connect(lambda: self.model_choose(custom=True))
        self.modelBoxG.addWidget(self.ModelChooseC, b0, 0, 1, 8)

        # compute segmentation w/ custom model
        self.ModelButtonC = QPushButton("run")
        self.ModelButtonC.setFont(self.medfont)
        self.ModelButtonC.setFixedWidth(35)
        self.ModelButtonC.clicked.connect(
            lambda: self.compute_segmentation(custom=True)
        )
        self.modelBoxG.addWidget(self.ModelButtonC, b0, 8, 1, 1)
        self.ModelButtonC.setEnabled(False)

        self.net_names = [
            "nuclei",
            "cyto2_cp3",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
            "cyto",
            "cyto2",
            "CPx",
        ]

        nett = [
            "nuclei",
            "cellpose (cyto2_cp3)",
            "tissuenet_cp3",
            "livecell_cp3",
            "yeast_PhC_cp3",
            "yeast_BF_cp3",
            "bact_phase_cp3",
            "bact_fluor_cp3",
            "deepbacs_cp3",
            "cyto",
            "cyto2",
            "CPx (from Cellpose2)",
        ]
        b0 += 1
        self.ModelChooseB = QComboBox()
        self.ModelChooseB.setFont(self.medfont)
        self.ModelChooseB.addItems(["dataset-specific models"])
        self.ModelChooseB.addItems(nett)
        self.ModelChooseB.setFixedWidth(175)
        tipstr = "dataset-specific models"
        self.ModelChooseB.setToolTip(tipstr)
        self.ModelChooseB.activated.connect(lambda: self.model_choose(custom=False))
        self.modelBoxG.addWidget(self.ModelChooseB, b0, 0, 1, 8)

        # compute segmentation w/ cp model
        self.ModelButtonB = QPushButton("run")
        self.ModelButtonB.setFont(self.medfont)
        self.ModelButtonB.setFixedWidth(35)
        self.ModelButtonB.clicked.connect(
            lambda: self.compute_segmentation(custom=False)
        )
        self.modelBoxG.addWidget(self.ModelButtonB, b0, 8, 1, 1)
        self.ModelButtonB.setEnabled(False)

        b += 1
        self.denoiseBox = QGroupBox("Image restoration")
        self.denoiseBox.setFont(self.boldfont)
        self.denoiseBoxG = QGridLayout()
        self.denoiseBox.setLayout(self.denoiseBoxG)
        self.l0.addWidget(self.denoiseBox, b, 0, 1, 9)

        b0 = 0

        # DENOISING
        self.DenoiseButtons = []
        nett = [
            "clear restore/filter",
            "filter image (settings below)",
            "denoise (please set cell diameter first)",
            "deblur (please set cell diameter first)",
            "upsample to 30. diameter (cyto3) or 17. diameter (nuclei) (please set cell diameter first) (disabled in 3D)",
            "one-click model trained to denoise+deblur+upsample (please set cell diameter first)",
        ]
        self.denoise_text = [
            "none",
            "filter",
            "denoise",
            "deblur",
            "upsample",
            "one-click",
        ]
        self.restore = None
        self.ratio = 1.0
        jj = 0
        w = 3
        for j in range(len(self.denoise_text)):
            self.DenoiseButtons.append(
                guiparts.DenoiseButton(self, self.denoise_text[j])
            )
            self.denoiseBoxG.addWidget(self.DenoiseButtons[-1], b0, jj, 1, w)
            self.DenoiseButtons[-1].setFixedWidth(75)
            self.DenoiseButtons[-1].setToolTip(nett[j])
            self.DenoiseButtons[-1].setFont(self.medfont)
            b0 += 1 if j % 2 == 1 else 0
            jj = 0 if j % 2 == 1 else jj + w

        # b0+=1
        self.save_norm = QCheckBox("save restored/filtered image")
        self.save_norm.setFont(self.medfont)
        self.save_norm.setToolTip("save restored/filtered image in _seg.npy file")
        self.save_norm.setChecked(True)
        # self.denoiseBoxG.addWidget(self.save_norm, b0, 0, 1, 8)

        b0 -= 3
        label = QLabel("restore-dataset:")
        label.setToolTip(
            "choose dataset and click [denoise], [deblur], [upsample], or [one-click]"
        )
        label.setFont(self.medfont)
        self.denoiseBoxG.addWidget(label, b0, 6, 1, 3)

        b0 += 1
        self.DenoiseChoose = QComboBox()
        self.DenoiseChoose.setFont(self.medfont)
        self.DenoiseChoose.addItems(["cyto3", "cyto2", "nuclei"])
        self.DenoiseChoose.setFixedWidth(85)
        tipstr = "choose model type and click [denoise], [deblur], or [upsample]"
        self.DenoiseChoose.setToolTip(tipstr)
        self.denoiseBoxG.addWidget(self.DenoiseChoose, b0, 6, 1, 3)

        b0 += 2
        # FILTERING
        self.filtBox = QCollapsible("custom filter settings")
        self.filtBox._toggle_btn.setFont(self.medfont)
        self.filtBoxG = QGridLayout()
        _content = QWidget()
        _content.setLayout(self.filtBoxG)
        _content.setMaximumHeight(0)
        _content.setMinimumHeight(0)
        # _content.layout().setContentsMargins(QtCore.QMargins(0, -20, -20, -20))
        self.filtBox.setContent(_content)
        self.denoiseBoxG.addWidget(self.filtBox, b0, 0, 1, 9)

        self.filt_vals = [0.0, 0.0, 0.0, 0.0]
        self.filt_edits = []
        labels = [
            "sharpen\nradius",
            "smooth\nradius",
            "tile_norm\nblocksize",
            "tile_norm\nsmooth3D",
        ]
        tooltips = [
            "set size of surround-subtraction filter for sharpening image",
            "set size of gaussian filter for smoothing image",
            "set size of tiles to use to normalize image",
            "set amount of smoothing of normalization values across planes",
        ]

        for p in range(4):
            label = QLabel(f"{labels[p]}:")
            label.setToolTip(tooltips[p])
            label.setFont(self.medfont)
            self.filtBoxG.addWidget(label, b0 + p // 2, 4 * (p % 2), 1, 2)
            self.filt_edits.append(QLineEdit())
            self.filt_edits[p].setText(str(self.filt_vals[p]))
            self.filt_edits[p].setFixedWidth(40)
            self.filt_edits[p].setFont(self.medfont)
            self.filtBoxG.addWidget(
                self.filt_edits[p], b0 + p // 2, 4 * (p % 2) + 2, 1, 2
            )
            self.filt_edits[p].setToolTip(tooltips[p])

        b0 += 3
        self.norm3D_cb = QCheckBox("norm3D")
        self.norm3D_cb.setFont(self.medfont)
        self.norm3D_cb.setChecked(True)
        self.norm3D_cb.setToolTip("run same normalization across planes")
        self.filtBoxG.addWidget(self.norm3D_cb, b0, 0, 1, 3)

        self.invert_cb = QCheckBox("invert")
        self.invert_cb.setFont(self.medfont)
        self.invert_cb.setToolTip("invert image")
        self.filtBoxG.addWidget(self.invert_cb, b0, 3, 1, 3)

        ## NEW
        b += 1
        b0 += 1
        self.MB = QGroupBox("Metrics")
        self.MB.setFont(self.boldfont)
        self.MB.setStyleSheet(
            "QGroupBox { border: 1px solid white; color:white; padding: 10px 0px;}"
        )
        self.MBg = QGridLayout()
        self.MB.setLayout(self.MBg)
        self.currentImageMask = ""
        self.indexCytoMask = -1
        self.indexNucleusMask = -1

        # select metrics to calculate
        self.calcSize = False
        self.SMCheckBox = QCheckBox("Area")
        self.SMCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.SMCheckBox.setFont(self.medfont)
        self.SMCheckBox.setChecked(False)
        self.SMCheckBox.setEnabled(False)
        self.SMCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Area of the cell in μm2"
        self.SMCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.SMCheckBox, 0, 0, 1, 7)

        self.calcRound = False
        self.RMCheckBox = QCheckBox("Roundness")
        self.RMCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.RMCheckBox.setFont(self.medfont)
        self.RMCheckBox.setChecked(False)
        self.RMCheckBox.setEnabled(False)
        self.RMCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Closer to 1 means more like a circle"
        self.RMCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.RMCheckBox, 0, 5, 1, 7)

        self.calcRatio = False
        self.RTCheckBox = QCheckBox("Ratio")
        self.RTCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.RTCheckBox.setFont(self.medfont)
        self.RTCheckBox.setChecked(False)
        self.RTCheckBox.setEnabled(False)
        self.RTCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Ratio between cyto and nucleus"
        self.RTCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.RTCheckBox, 1, 0, 1, 7)

        self.calcVoronoi = False
        self.VDCheckBox = QCheckBox("Voronoi")
        self.VDCheckBox.setStyleSheet("color: rgb(190,190,190);")
        self.VDCheckBox.setFont(self.medfont)
        self.VDCheckBox.setChecked(False)
        self.VDCheckBox.setEnabled(False)
        self.VDCheckBox.toggled.connect(self.toggle_masks)
        tipstr = "Ratio between cyto and nucleus"
        self.VDCheckBox.setToolTip(tipstr)
        self.MBg.addWidget(self.VDCheckBox, 1, 5, 1, 7)

        # calculate the selected metrics
        self.CalculateButton = QPushButton("calculate")
        self.CalculateButton.clicked.connect(
            lambda: self.features_class.calculate_metrics(self)
        )
        self.MBg.addWidget(self.CalculateButton, 0, 10, 1, 2)
        self.CalculateButton.setEnabled(False)
        # self.CalculateButton.setStyleSheet(self.styleInactive)

        self.l0.addWidget(self.MB, b, 0, 1, 9)
        ##

        b += 1
        self.l0.addWidget(QLabel(""), b, 0, 1, 9)
        self.l0.setRowStretch(b, 100)

        b += 1
        # scale toggle
        self.scale_on = True
        self.ScaleOn = QCheckBox("scale disk on")
        self.ScaleOn.setFont(self.medfont)
        self.ScaleOn.setStyleSheet("color: rgb(150,50,150);")
        self.ScaleOn.setChecked(True)
        self.ScaleOn.setToolTip("see current diameter as red disk at bottom")
        self.ScaleOn.toggled.connect(self.toggle_scale)
        self.l0.addWidget(self.ScaleOn, b, 0, 1, 5)

        return b

    def update_px_to_mm(self):
        """
        Update the pixel-to-millimeter conversion factor.

            This method retrieves the current value from a user interface element,
            representing the conversion ratio from pixels to millimeters, and updates
            the instance variable accordingly.

            Attributes:
                self.px_to_mm: The conversion factor from pixels to millimeters as a float.

            Returns:
                None
        """
        self.px_to_mm = float(self.pixTomicro.text())

    def level_change(self, r):
        """
        Update the saturation levels based on the slider's value.

            This method retrieves the current value from the specified slider and updates
            the saturation level for that channel. If the auto-button is not checked, it
            synchronizes the saturation levels across all indices of the specified channel.
            Finally, it updates the plot to reflect the changes.

            Args:
                r: The index representing the color channel (0 for red, 1 for green, 2 for blue).

            Returns:
                None
        """
        r = ["red", "green", "blue"].index(r)
        if self.loaded:
            sval = self.sliders[r].value()
            self.saturation[r][self.currentZ] = sval
            if not self.autobtn.isChecked():
                for r in range(3):
                    for i in range(len(self.saturation[r])):
                        self.saturation[r][i] = self.saturation[r][self.currentZ]
            self.update_plot()

    def keyPressEvent(self, event):
        """
        Handles key press events for updating the user interface and triggering actions.

            This method listens for key press events and performs various actions based on the specific key pressed.
            It can toggle UI elements, navigate through images, change colors and brush settings, and update the plot.

            Args:
                event: The key press event containing information about the pressed key and any modifiers.

            Returns:
                None
        """
        if self.loaded:
            if not (
                event.modifiers()
                & (
                    QtCore.Qt.ControlModifier
                    | QtCore.Qt.ShiftModifier
                    | QtCore.Qt.AltModifier
                )
                or self.in_stroke
            ):
                updated = False
                if len(self.current_point_set) > 0:
                    if event.key() == QtCore.Qt.Key_Return:
                        self.add_set()
                else:
                    nviews = self.ViewDropDown.count() - 1
                    nviews += int(
                        self.ViewDropDown.model()
                        .item(self.ViewDropDown.count() - 1)
                        .isEnabled()
                    )
                    if event.key() == QtCore.Qt.Key_X:
                        self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Z:
                        self.OCheckBox.toggle()
                    if (
                        event.key() == QtCore.Qt.Key_Left
                        or event.key() == QtCore.Qt.Key_A
                    ):
                        self.get_prev_image()
                    elif (
                        event.key() == QtCore.Qt.Key_Right
                        or event.key() == QtCore.Qt.Key_D
                    ):
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_PageDown:
                        self.view = (self.view + 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)
                    elif event.key() == QtCore.Qt.Key_PageUp:
                        self.view = (self.view - 1) % (nviews)
                        self.ViewDropDown.setCurrentIndex(self.view)

                # can change background or stroke size if cell not finished
                if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_W:
                    self.color = (self.color - 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (
                    event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_S
                ):
                    self.color = (self.color + 1) % (6)
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_R:
                    if self.color != 1:
                        self.color = 1
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_G:
                    if self.color != 2:
                        self.color = 2
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif event.key() == QtCore.Qt.Key_B:
                    if self.color != 3:
                        self.color = 3
                    else:
                        self.color = 0
                    self.RGBDropDown.setCurrentIndex(self.color)
                elif (
                    event.key() == QtCore.Qt.Key_Comma
                    or event.key() == QtCore.Qt.Key_Period
                ):
                    count = self.BrushChoose.count()
                    gci = self.BrushChoose.currentIndex()
                    if event.key() == QtCore.Qt.Key_Comma:
                        gci = max(0, gci - 1)
                    else:
                        gci = min(count - 1, gci + 1)
                    self.BrushChoose.setCurrentIndex(gci)
                    self.brush_choose()
                if not updated:
                    self.update_plot()
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Equal:
            self.p0.keyPressEvent(event)

    def autosave_on(self):
        """
        Sets the autosave feature based on the state of a checkbox.

            This method checks the state of the SCheckBox widget. If the checkbox
            is checked, it enables the autosave feature; if unchecked, it disables it.

            Parameters:
                None

            Returns:
                None
        """
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        """
        Check and update the GPU usage status.

            This method checks whether a GPU is available for use with
            the application and updates the corresponding UI components
            accordingly. If a GPU is detected and can be used with Torch,
            the checkbox for GPU usage is enabled and checked. Otherwise,
            the checkbox is disabled and styled to indicate that GPU usage
            is not available.

            Parameters:
                None

            Returns:
                None
        """
        # also decide whether or not to use torch
        self.useGPU.setChecked(False)
        self.useGPU.setEnabled(False)
        if core.use_gpu(use_torch=True):
            self.useGPU.setEnabled(True)
            self.useGPU.setChecked(True)
        else:
            self.useGPU.setStyleSheet("color: rgb(80,80,80);")

    def get_channels(self):
        """
        Retrieve the current indices of the selected channels.

            This method checks the currently selected channels in the GUI and adjusts them
            based on the current model and other constraints. It ensures that valid channel
            selections are made, particularly when the number of available channels is limited.

            Returns:
                A list of integers representing the indices of the selected channels.
        """
        channels = [
            self.ChannelChoose[0].currentIndex(),
            self.ChannelChoose[1].currentIndex(),
        ]
        if hasattr(self, "current_model"):
            if self.current_model == "nuclei":
                channels[1] = 0
        if channels[0] == 0:
            channels[1] = 0
        if self.nchan == 1:
            channels = [0, 0]
        elif self.nchan == 2:
            if channels[0] == 3:
                channels[0] = 1 if channels[1] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
            if channels[1] == 3:
                channels[1] = 1 if channels[0] != 1 else 2
                print(
                    f"GUI_WARNING: only two channels in image, cannot use blue channel, changing channels"
                )
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        return channels

    def model_choose(self, custom=False):
        """
        Select and load a machine learning model based on user input.

            This method retrieves the currently selected model from either a custom
            model chooser or a default chooser based on the provided parameters. It
            then initializes the model, updates the diameter label in the GUI, and
            prints relevant information for user feedback.

            Args:
                custom: A boolean flag indicating whether to use the custom model selector.

            Returns:
                None: This method does not return a value.
        """
        index = (
            self.ModelChooseC.currentIndex()
            if custom
            else self.ModelChooseB.currentIndex()
        )
        if index > 0:
            if custom:
                model_name = self.ModelChooseC.currentText()
            else:
                model_name = self.net_names[index - 1]
            print(f"GUI_INFO: selected model {model_name}, loading now")
            self.initialize_model(model_name=model_name, custom=custom)
            self.diameter = self.model.diam_labels
            self.Diameter.setText("%0.2f" % self.diameter)
            print(
                f"GUI_INFO: diameter set to {self.diameter: 0.2f} (but can be changed)"
            )

    def calibrate_size(self):
        """
        Calibrate the size of cells in the current stack.

            This method estimates the diameter of cells using a predefined model,
            updates the user interface with the estimated diameter, and sets the
            progress to complete.

            Parameters:
                None

            Returns:
                None
        """
        self.initialize_model(model_name="cyto3")
        diams, _ = self.model.sz.eval(
            self.stack[self.currentZ].copy(),
            channels=self.get_channels(),
            progress=self.progress,
        )
        diams = np.maximum(5.0, diams)
        self.logger.info(
            "estimated diameter of cells using %s model = %0.1f pixels"
            % (self.current_model, diams)
        )
        self.Diameter.setText("%0.1f" % diams)
        self.diameter = diams
        self.update_scale()
        self.progress.setValue(100)

    def toggle_scale(self):
        """
        Toggle the visibility of the scale in the plot.

            This method adds or removes the scale item from the plot based on its current state.
            If the scale is currently displayed, it will be removed, and if it is not displayed, it will be added.

            Parameters:
                None

            Returns:
                None
        """
        if self.scale_on:
            self.p0.removeItem(self.scale)
            self.scale_on = False
        else:
            self.p0.addItem(self.scale)
            self.scale_on = True

    def enable_buttons(self):
        """
        Enables various UI buttons and updates the plot based on the current state.

            This method activates buttons related to model selection, style options, and denoising, as well as sliders for settings.
            It also updates the window title and toggles certain operations depending on the loaded data.
            Specific buttons are disabled based on the application's context, such as 3D loading.

            Attributes:
                self.model_strings: The list containing model options.
                self.StyleButtons: A collection of style option buttons.
                self.DenoiseButtons: A collection of buttons for denoising options.
                self.load_3D: A boolean indicating if 3D loading is enabled.
                self.sliders: A list of slider controls for different channels.
                self.nchan: The number of channels affecting the enabled state of sliders.
                self.filename: The current filename displayed in the window title.

            Returns:
                None
        """
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(True)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(True)
        for i in range(len(self.DenoiseButtons)):
            self.DenoiseButtons[i].setEnabled(True)
        if self.load_3D:
            self.DenoiseButtons[-2].setEnabled(False)
        self.ModelButtonB.setEnabled(True)
        self.SizeButton.setEnabled(True)
        self.newmodel.setEnabled(True)
        self.loadMasks.setEnabled(True)
        self.keepMask.setEnabled(False)  # New
        self.saveMasks.setEnabled(False)  # New

        for n in range(self.nchan):
            self.sliders[n].setEnabled(True)
        for n in range(self.nchan, 3):
            self.sliders[n].setEnabled(True)

        self.toggle_mask_ops()

        self.update_plot()
        self.setWindowTitle(self.filename)

    def disable_buttons_removeROIs(self):
        """
        Disable various buttons in the user interface when removing ROIs.

            This method disables a set of buttons related to model operations and style selections
            when the condition to remove ROIs is triggered. If model strings are present, it ensures
            that certain model-related buttons are also disabled, thus preventing user interaction
            while performing the deletion process.

            Parameters:
                None

            Returns:
                None
        """
        if len(self.model_strings) > 0:
            self.ModelButtonC.setEnabled(False)
        for i in range(len(self.StyleButtons)):
            self.StyleButtons[i].setEnabled(False)
        self.ModelButtonB.setEnabled(False)
        self.SizeButton.setEnabled(False)
        self.newmodel.setEnabled(False)
        self.loadMasks.setEnabled(False)
        self.saveSet.setEnabled(False)
        self.savePNG.setEnabled(False)
        self.saveFlows.setEnabled(False)
        self.saveOutlines.setEnabled(False)
        self.saveROIs.setEnabled(False)

        self.MakeDeletionRegionButton.setEnabled(False)
        self.DeleteMultipleROIButton.setEnabled(False)
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)

    def toggle_mask_ops(self):
        """
        Toggle various masking operations for the layer.

            This method updates the current layer and toggles the states of
            saving and removal operations associated with it.

            Parameters:
                None

            Returns:
                None
        """
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()

    def toggle_saving(self):
        """
        Toggle the enabled state of the saving options based on the current cell count.

            This method enables or disables various saving options within the application
            depending on whether there are any cells present. If the number of cells is greater
            than zero, the saving options will be enabled; otherwise, they will be disabled.

            Attributes:
                self.ncells: The current number of cells. When greater than zero, saving options
                are enabled; otherwise, they are disabled.

            Returns:
                None: This method does not return any value.
        """
        if self.ncells > 0:
            self.saveSet.setEnabled(True)
            self.savePNG.setEnabled(True)
            self.saveFlows.setEnabled(True)
            self.saveOutlines.setEnabled(True)
            self.saveROIs.setEnabled(True)
        else:
            self.saveSet.setEnabled(False)
            self.savePNG.setEnabled(False)
            self.saveFlows.setEnabled(False)
            self.saveOutlines.setEnabled(False)
            self.saveROIs.setEnabled(False)

    def toggle_removals(self):
        """
        Toggle the state of removal-related buttons based on the number of cells.

            This method enables or disables various buttons related to removal actions
            based on whether the number of cells is greater than zero. If there are cells
            present, all related buttons are activated; otherwise, they are deactivated.

            Parameters:
                None

            Returns:
                None
        """
        if self.ncells > 0:
            self.ClearButton.setEnabled(True)
            self.remcell.setEnabled(True)
            self.undo.setEnabled(True)
            self.MakeDeletionRegionButton.setEnabled(True)
            self.DeleteMultipleROIButton.setEnabled(True)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)
        else:
            self.ClearButton.setEnabled(False)
            self.remcell.setEnabled(False)
            self.undo.setEnabled(False)
            self.MakeDeletionRegionButton.setEnabled(False)
            self.DeleteMultipleROIButton.setEnabled(False)
            self.DoneDeleteMultipleROIButton.setEnabled(False)
            self.CancelDeleteMultipleROIButton.setEnabled(False)

    def remove_action(self):
        """
        Remove the currently selected action.

            This method removes the action corresponding to the currently selected item.
            It checks if there is a selection before attempting to remove it.

            Parameters:
                None

            Returns:
                None
        """
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        """
        Undoes the last action performed.

            This method removes the most recent stroke if it matches the current Z coordinate.
            If there are no strokes matching the current Z, it will remove the last cell if any cells exist.

            Parameters:
                None

            Returns:
                None
        """
        if len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ:
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        """
        Reverses the most recent cell removal action.

            This method restores the last removed cell, allowing the user to undo
            the removal action performed previously.

            Parameters:
                None

            Returns:
                None
        """
        self.undo_remove_cell()

    def get_files(self):
        """
        Retrieve image files from a specified folder and find the index of the current file.

            This method fetches image files from a folder based on a predefined mask filter.
            It identifies the current file's name and determines its index within the retrieved
            list of image files.

            Returns:
                A tuple containing:
                    - A list of image file paths that match the mask filter.
                    - An integer representing the index of the current file within the list of image files.
        """
        folder = os.path.dirname(self.filename)
        mask_filter = "_masks"
        images = get_image_files(folder, mask_filter)
        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]
        idx = np.nonzero(np.array(fnames) == f0)[0][0]
        return images, idx

    def get_prev_image(self):
        """
        Retrieve the previous image in a cyclic manner.

            This method loads the image that precedes the currently active image
            in a circular sequence. If the current image is the first in the list,
            it wraps around to the last image.

            Parameters:
                None

            Returns:
                The loaded image from the previous index in the image list.
        """
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        """
        Retrieve the next image from the list of available images.

            This method loads the next image based on the current index,
            cycling back to the start if the end of the list is reached.

            Args:
                load_seg: A flag indicating whether to load the associated segmentation
                    of the image. Defaults to True.

            Returns:
                The next image after loading it from the specified file.
        """
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        """
        Handle drag enter events for a widget.

            This method is triggered when a drag operation enters the widget. It checks
            whether the data being dragged contains URLs. If so, it accepts the event,
            allowing the drag operation to continue. Otherwise, it ignores the event.

            Args:
                event: The event object containing information about the drag operation.

            Returns:
                None
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handle the drop event for loading files.

            This method processes a drop event to load either a segmentation file or an image file
            based on the file type of the dropped item. If the dropped file is a .npy file,
            it loads the segmentation data; otherwise, it loads the image data.

            Args:
                event: The drop event containing the mime data with URLs of the dropped files.

            Returns:
                None
        """
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if os.path.splitext(files[0])[-1] == ".npy":
            io._load_seg(self, filename=files[0], load_3D=self.load_3D)
        else:
            io._load_image(self, filename=files[0], load_seg=True, load_3D=self.load_3D)

    def toggle_masks(self):
        """
        Toggle the visibility of masks and related calculations.

            This method checks the status of various checkboxes to update the
            attributes that control whether masks, outlines, and calculations
            for size, roundness, ratio, and Voronoi are enabled or disabled.
            If neither masks nor outlines are enabled, it removes the layer
            from the view; otherwise, it adds the layer back and redraws it.
            Furthermore, if the data is loaded, the plot and layer are updated.

            Attributes updated:
                - masksOn: Indicates if masks are enabled.
                - outlinesOn: Indicates if outlines are enabled.
                - calcSize: Indicates if size calculation is enabled.
                - calcRound: Indicates if roundness calculation is enabled.
                - calcRatio: Indicates if ratio calculation is enabled.
                - calcVoronoi: Indicates if Voronoi calculation is enabled.

            Returns:
                None
        """
        if self.MCheckBox.isChecked():
            self.masksOn = True
        else:
            self.masksOn = False

        if self.OCheckBox.isChecked():
            self.outlinesOn = True
        else:
            self.outlinesOn = False

        if self.SMCheckBox.isChecked():
            self.calcSize = True
        else:
            self.calcSize = False

        if self.RMCheckBox.isChecked():
            self.calcRound = True
        else:
            self.calcRound = False

        if self.RTCheckBox.isChecked():
            self.calcRatio = True
        else:
            self.calcRatio = False

        if self.VDCheckBox.isChecked():
            self.calcVoronoi = True
        else:
            self.calcVoronoi = False

        if not self.masksOn and not self.outlinesOn:
            self.p0.removeItem(self.layer)
            self.layer_off = True
        else:
            if self.layer_off:
                self.p0.addItem(self.layer)
            self.draw_layer()
            self.update_layer()

        if self.loaded:
            self.update_plot()
            self.update_layer()

    def make_viewbox(self):
        """
        Creates and configures a ViewBox for displaying images.

            This method initializes a ViewBox component with specific settings,
            including cursor type, aspect ratio locking, and mouse interactivity.
            It adds several graphical items to the ViewBox, including an image
            item and a drawing layer, and sets their properties.

            Parameters:
                None

            Returns:
                None
        """
        self.p0 = guiparts.ViewBoxNoRightDrag(
            parent=self,
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True,
        )
        self.p0.setCursor(QtCore.Qt.CrossCursor)
        self.brush_size = 3
        self.win.addItem(self.p0, 0, 0, rowspan=1, colspan=1)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = guiparts.ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0, 255])
        self.scale = pg.ImageItem(viewbox=self.p0, parent=self)
        self.scale.setLevels([0, 255])
        self.p0.scene().contextMenuItem = self.p0
        # self.p0.setMouseEnabled(x=False,y=False)
        self.Ly, self.Lx = 512, 512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)
        self.p0.addItem(self.scale)

    def reset(self):
        """
        Resets the state of the object to its initial configuration.

            This method clears and reinitializes various properties of the object,
            effectively restoring it to a default state. This includes setting
            parameters related to point sets, image stack, and menus.

            Parameters:
                None

            Returns:
                None
        """
        # ---- start sets of points ---- #
        self.selected = 0
        self.nchan = 3
        self.loaded = False
        self.channel = [0, 1]
        self.current_point_set = []
        self.in_stroke = False
        self.strokes = []
        self.stroke_appended = True
        self.resize = False
        self.ncells = 0
        self.zdraw = []
        self.removed_cell = []
        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]

        # -- zero out image stack -- #
        self.opacity = 128  # how opaque masks should be
        self.outcolor = [200, 200, 255, 200]
        self.NZ, self.Ly, self.Lx = 1, 224, 224
        self.saturation = []
        for r in range(3):
            self.saturation.append([[0, 255] for n in range(self.NZ)])
            self.sliders[r].setValue([0, 255])
            self.sliders[r].setEnabled(False)
            self.sliders[r].show()
        self.currentZ = 0
        self.flows = [[], [], [], [], [[]]]
        # masks matrix
        # image matrix with a scale disk
        self.stack = np.zeros((1, self.Ly, self.Lx, 3))
        self.Lyr, self.Lxr = self.Ly, self.Lx
        self.Ly0, self.Lx0 = self.Ly, self.Lx
        self.radii = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
        self.cellpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        self.outpix = np.zeros((1, self.Ly, self.Lx), np.uint16)
        if self.restore and "upsample" in self.restore:
            self.cellpix_resize = self.cellpix
            self.cellpix_orig = self.cellpix
            self.outpix_resize = self.cellpix
            self.outpix_orig = self.cellpix
        self.ismanual = np.zeros(0, "bool")

        # -- set menus to default -- #
        self.color = 0
        self.RGBDropDown.setCurrentIndex(self.color)
        self.view = 0
        self.ViewDropDown.setCurrentIndex(0)
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        self.delete_restore()

        self.clear_all()

        # self.update_plot()
        self.filename = []
        self.loaded = False
        self.recompute_masks = False

        self.deleting_multiple = False
        self.removing_cells_list = []
        self.removing_region = False
        self.remove_roi_obj = None

    def delete_restore(self):
        """delete restored imgs but don't reset settings"""
        if hasattr(self, "stack_filtered"):
            del self.stack_filtered
        if hasattr(self, "cellpix_orig"):
            self.cellpix = self.cellpix_orig.copy()
            self.outpix = self.outpix_orig.copy()
            del self.outpix_orig, self.outpix_resize
            del self.cellpix_orig, self.cellpix_resize

    def clear_restore(self):
        """delete restored imgs and reset settings"""
        print("GUI_INFO: clearing restored image")
        self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(False)
        if self.ViewDropDown.currentIndex() == self.ViewDropDown.count() - 1:
            self.ViewDropDown.setCurrentIndex(0)
        self.delete_restore()
        self.restore = None
        self.ratio = 1.0
        self.set_normalize_params(self.get_normalize_params())

    def brush_choose(self):
        """
        Selects the brush size based on the current index of the brush chooser.

            This method updates the brush size by calculating it from the current index of the brush chooser widget.
            If a layer is loaded, it adjusts the drawing kernel of the layer to match the new brush size
            and triggers an update for the layer.

            Attributes:
                brush_size: The size of the brush calculated from the current index.

            Returns:
                None
        """
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        """
        Clears all relevant states and resets pixel data structures.

            This method resets the internal state of the object by clearing previous selections,
            initializing pixel data structures to empty states, and updating relevant layers and scales.
            It configures different sizes based on whether restoration through upsampling is required.

            Parameters:
                None

            Returns:
                None
        """
        self.prev_selected = 0
        self.selected = 0
        if self.restore and "upsample" in self.restore:
            self.layerz = 0 * np.ones((self.Lyr, self.Lxr, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Lyr, self.Lxr), np.uint16)
            self.cellpix_resize = self.cellpix.copy()
            self.outpix_resize = self.outpix.copy()
            self.cellpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
            self.outpix_orig = np.zeros((self.NZ, self.Ly0, self.Lx0), np.uint16)
        else:
            self.layerz = 0 * np.ones((self.Ly, self.Lx, 4), np.uint8)
            self.cellpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)
            self.outpix = np.zeros((self.NZ, self.Ly, self.Lx), np.uint16)

        self.cellcolors = np.array([255, 255, 255])[np.newaxis, :]
        self.ncells = 0
        self.toggle_removals()
        self.update_scale()
        self.update_layer()

    def select_cell(self, idx):
        """
        Selects a cell and processes its associated pixel data.

            This method updates the current selection index for a cell, processes the pixel data associated with the selected cell,
            creates a mask, removes small labels, and performs measurements on the selected cell's characteristics.
            Finally, it updates the associated graphical layer to reflect the changes.

            Args:
                idx: The index of the cell to be selected.

            Returns:
                None
        """
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected > 0:
            np.set_printoptions(threshold=sys.maxsize)

            slices = find_objects(self.cellpix[0].astype(int))
            si = slices[self.selected - 1]
            sr, sc = si
            # mask = (self.cellpix[0][sr, sc] == (self.selected)).astype(np.uint8)
            tmp_cellpix = np.copy(self.cellpix[0])
            tmp_cellpix[self.selected != self.cellpix[0]] = 0
            tmp_cellpix[self.selected == self.cellpix[0]] = 255

            # mask_shape = mask.shape
            # for i in range(0, mask_shape[0]):
            #     for j in range(0, mask_shape[1]):
            #         mask[i][j] = 255 if mask[i][j] > 0 else 0

            mask = tmp_cellpix.astype(np.uint8)

            mask = np.pad(mask, 1, mode="constant")
            im = Image.fromarray(mask)
            im.save("hola1.jpg")

            Zlabeled, Nlabels = ndimage.label(mask)
            label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
            for label, size in enumerate(label_size):
                print("label %s is %s pixels in size" % (label, size))

            # now remove the labels
            for label, size in enumerate(label_size):
                if size < 5:
                    mask[Zlabeled == label] = 0

            im = Image.fromarray(mask)
            im.save("hola2.jpg")

            labels = dip.Label(mask[:, :] > 0)
            msr = dip.MeasurementTool.Measure(
                labels,
                features=[
                    "Perimeter",
                    "SolidArea",
                    "Roundness",
                    "Circularity",
                    "Center",
                ],
            )
            print(msr)
            print("IDX: ", self.selected)
            print("Size in px: ", msr[1]["SolidArea"][0])
            print(
                "Size in μm: ", round(msr[1]["SolidArea"][0] * pow(self.px_to_mm, 2), 2)
            )

            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def select_cell_multi(self, idx):
        """
        Selects multiple cells based on the provided index.

            This method updates the graphical layer for cells that match the
            specified index, changing their appearance to indicate selection
            with a white color and the specified opacity.

            Args:
                idx: The index of the cells to select. Only cells corresponding
                      to this index will be updated in the graphical layer.

            Returns:
                None: This method does not return a value.
        """
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def unselect_cell(self):
        """
        Unselects the currently selected cell and updates the layer representation.

            This method clears the selection of the cell and updates the display
            layers accordingly. It modifies the color and opacity of the cell that was
            previously selected based on its index. If outlines are enabled, it also adjusts
            the outline color for the unselected cell.

            Parameters:
                None

            Returns:
                None
        """
        if self.selected > 0:
            idx = self.selected
            if idx < self.ncells + 1:
                z = self.currentZ
                self.layerz[self.cellpix[z] == idx] = np.append(
                    self.cellcolors[idx], self.opacity
                )
                if self.outlinesOn:
                    self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                        np.uint8
                    )
                    # [0,0,0,self.opacity])
                self.update_layer()
        self.selected = 0

    def unselect_cell_multi(self, idx):
        """
        Unselects multiple cells in the current layer.

            This method updates the layer by removing the selection of cells identified
            by the given index. It modifies the colors and outlines of the selected cells
            according to the specified properties and updates the layer afterwards.

            Args:
                idx: An index representing the cells to unselect.

            Returns:
                None: This method does not return a value, but updates the layer state.
        """
        z = self.currentZ
        self.layerz[self.cellpix[z] == idx] = np.append(
            self.cellcolors[idx], self.opacity
        )
        if self.outlinesOn:
            self.layerz[self.outpix[z] == idx] = np.array(self.outcolor).astype(
                np.uint8
            )
            # [0,0,0,self.opacity])
        self.update_layer()

    def remove_cell(self, idx):
        """
        Removes specified cells from the data structure.

            This method updates the state of the cellpix and outpix arrays
            by removing the cells identified by the given index or indices.
            It ensures that cells are removed in reverse order to maintain
            the integrity of the indexing.

            Args:
                idx: The index or indices of the cells to be removed.
                      This can be a single integer or a list of integers.

            Returns:
                None: This method does not return a value but updates the internal state
                of the object by modifying the number of cells and possibly
                the buttons for UI interactions.
        """
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        # because the function remove_single_cell updates the state of the cellpix and outpix arrays
        # by reindexing cells to avoid gaps in the indices, we need to remove the cells in reverse order
        # so that the indices are correct
        idx.sort(reverse=True)
        for i in idx:
            self.remove_single_cell(i)
        self.ncells -= len(idx)  # _save_sets uses ncells

        if self.ncells == 0:
            self.ClearButton.setEnabled(False)
        if self.NZ == 1:
            io._save_sets_with_check(self)

        self.update_layer()

    def remove_single_cell(self, idx):
        """
        Removes a single cell from the selection and updates related data structures.

            This method removes the specified cell index from the internal representations of cell
            and output pixels. It updates the necessary masks and adjusts other pixel indices accordingly.
            If there is only one z-layer, it stores information about the removed cell for tracking changes.

            Args:
                idx: The index of the cell to be removed from the selection.

            Returns:
                None: This method modifies the state of the object but does not return a value.
        """
        # remove from manual array
        self.selected = 0
        if self.NZ > 1:
            zextent = ((self.cellpix == idx).sum(axis=(1, 2)) > 0).nonzero()[0]
        else:
            zextent = [0]
        for z in zextent:
            cp = self.cellpix[z] == idx
            op = self.outpix[z] == idx
            # remove from self.cellpix and self.outpix
            self.cellpix[z, cp] = 0
            self.outpix[z, op] = 0
            if z == self.currentZ:
                # remove from mask layer
                self.layerz[cp] = np.array([0, 0, 0, 0])

        # reduce other pixels by -1
        self.cellpix[self.cellpix > idx] -= 1
        self.outpix[self.outpix > idx] -= 1

        if self.NZ == 1:
            self.removed_cell = [
                self.ismanual[idx - 1],
                self.cellcolors[idx],
                np.nonzero(cp),
                np.nonzero(op),
            ]
            self.redo.setEnabled(True)
            ar, ac = self.removed_cell[2]
            d = datetime.datetime.now()
            self.track_changes.append(
                [d.strftime("%m/%d/%Y, %H:%M:%S"), "removed mask", [ar, ac]]
            )
        # remove cell from lists
        self.ismanual = np.delete(self.ismanual, idx - 1)
        self.cellcolors = np.delete(self.cellcolors, [idx], axis=0)
        del self.zdraw[idx - 1]
        print("GUI_INFO: removed cell %d" % (idx - 1))

    def remove_region_cells(self):
        """
        Removes cells from the current selection and creates a new ROI in the center of the view.

            This method clears the currently selected cells, disables the buttons related to removing ROIs,
            and initializes a new region of interest (ROI) in the middle of the view. The ROI is sized to be
            half the dimensions of the current view.

            Parameters:
                None

            Returns:
                None
        """
        if self.removing_cells_list:
            for idx in self.removing_cells_list:
                self.unselect_cell_multi(idx)
            self.removing_cells_list.clear()
        self.disable_buttons_removeROIs()
        self.removing_region = True

        self.clear_multi_selected_cells()

        # make roi region here in center of view, making ROI half the size of the view
        roi_width = self.p0.viewRect().width() / 2
        x_loc = self.p0.viewRect().x() + (roi_width / 2)
        roi_height = self.p0.viewRect().height() / 2
        y_loc = self.p0.viewRect().y() + (roi_height / 2)

        pos = [x_loc, y_loc]
        roi = pg.RectROI(
            pos, [roi_width, roi_height], pen=pg.mkPen("y", width=2), removable=True
        )
        roi.sigRemoveRequested.connect(self.remove_roi)
        roi.sigRegionChangeFinished.connect(self.roi_changed)
        self.p0.addItem(roi)
        self.remove_roi_obj = roi
        self.roi_changed(roi)

    def delete_multiple_cells(self):
        """
        Delete multiple cells from the application interface.

            This method prepares the application to delete multiple cells by
            unselecting any selected cell and enabling the relevant buttons
            for confirming the deletion process. It also sets an internal
            flag to indicate that multiple deletions are being processed.

            Attributes:
                None

            Returns:
                None
        """
        self.unselect_cell()
        self.disable_buttons_removeROIs()
        self.DoneDeleteMultipleROIButton.setEnabled(True)
        self.MakeDeletionRegionButton.setEnabled(True)
        self.CancelDeleteMultipleROIButton.setEnabled(True)
        self.deleting_multiple = True

    def done_remove_multiple_cells(self):
        """
        Finalizes the removal of multiple cells and updates the UI.

            This method is called to complete the process of removing multiple cells
            from the graphical interface. It disables certain buttons related to the
            deletion process and processes the list of cells to be removed. It also
            handles the removal of any associated regions of interest (ROIs).

            Parameters:
                None

            Returns:
                None
        """
        self.deleting_multiple = False
        self.removing_region = False
        self.DoneDeleteMultipleROIButton.setEnabled(False)
        self.MakeDeletionRegionButton.setEnabled(False)
        self.CancelDeleteMultipleROIButton.setEnabled(False)

        if self.removing_cells_list:
            self.removing_cells_list = list(set(self.removing_cells_list))
            display_remove_list = [i - 1 for i in self.removing_cells_list]
            print(f"GUI_INFO: removing cells: {display_remove_list}")
            self.remove_cell(self.removing_cells_list)
            self.removing_cells_list.clear()
            self.unselect_cell()
        self.enable_buttons()

        if self.remove_roi_obj is not None:
            self.remove_roi(self.remove_roi_obj)

    def merge_cells(self, idx):
        """
        Merge two cells based on the selected index.

            This method updates the state of the currently selected cell and merges it with
            a previously selected cell if they are adjacent. It handles the graphical representation
            of the cells by drawing masks and updating the layers accordingly.

            Args:
                idx: The index of the cell to merge with the previously selected cell.

            Returns:
                None: This method does not return a value. It modifies the internal state and the
                graphical representation of the cells.
        """
        self.prev_selected = self.selected
        self.selected = idx
        if self.selected != self.prev_selected:
            for z in range(self.NZ):
                ar0, ac0 = np.nonzero(self.cellpix[z] == self.prev_selected)
                ar1, ac1 = np.nonzero(self.cellpix[z] == self.selected)
                touching = np.logical_and(
                    (ar0[:, np.newaxis] - ar1) < 3, (ac0[:, np.newaxis] - ac1) < 3
                ).sum()
                ar = np.hstack((ar0, ar1))
                ac = np.hstack((ac0, ac1))
                vr0, vc0 = np.nonzero(self.outpix[z] == self.prev_selected)
                vr1, vc1 = np.nonzero(self.outpix[z] == self.selected)
                self.outpix[z, vr0, vc0] = 0
                self.outpix[z, vr1, vc1] = 0
                if touching > 0:
                    mask = np.zeros((np.ptp(ar) + 4, np.ptp(ac) + 4), np.uint8)
                    mask[ar - ar.min() + 2, ac - ac.min() + 2] = 1
                    contours = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )
                    pvc, pvr = contours[-2][0].squeeze().T
                    vr, vc = pvr + ar.min() - 2, pvc + ac.min() - 2

                else:
                    vr = np.hstack((vr0, vr1))
                    vc = np.hstack((vc0, vc1))
                color = self.cellcolors[self.prev_selected]
                self.draw_mask(z, ar, ac, vr, vc, color, idx=self.prev_selected)
            self.remove_cell(self.selected)
            print("GUI_INFO: merged two cells")
            self.update_layer()
            io._save_sets_with_check(self)
            self.undo.setEnabled(False)
            self.redo.setEnabled(False)

    def undo_remove_cell(self):
        """
        Restores a previously removed cell to the drawing layer.

            This method checks if there are any cells in the removed_cell queue. If a cell is present, it retrieves the necessary attributes (such as position and color) to redraw the cell on the canvas. The cell is then added back to the internal state, including its color and manual status, and updates the layer. It also resets the removed_cell queue and disables the redo option.

            There are no parameters for this method.

            Returns:
                None: This method does not return a value.
        """
        if len(self.removed_cell) > 0:
            z = 0
            ar, ac = self.removed_cell[2]
            vr, vc = self.removed_cell[3]
            color = self.removed_cell[1]
            self.draw_mask(z, ar, ac, vr, vc, color)
            self.toggle_mask_ops()
            self.cellcolors = np.append(self.cellcolors, color[np.newaxis, :], axis=0)
            self.ncells += 1
            self.ismanual = np.append(self.ismanual, self.removed_cell[0])
            self.zdraw.append([])
            print(">>> added back removed cell")
            self.update_layer()
            io._save_sets_with_check(self)
            self.removed_cell = []
            self.redo.setEnabled(False)

    def remove_stroke(self, delete_points=True, stroke_ind=-1):
        """
        Removes a stroke from the layer and updates the relevant state.

        This method deletes a specified stroke from the current layer,
        updates the pixel information, and optionally removes the stroke
        from the set of current points. It modifies the appearance of the
        layer based on selected options such as outlines and mask visibility.

        Args:
            e_points: A flag indicating whether to remove end points or not.
            stroke_ind: The index of the stroke to remove.

        Returns:
            None: This method does not return a value but modifies the internal state.
        """
        stroke = np.array(self.strokes[stroke_ind])
        cZ = self.currentZ
        inZ = stroke[0, 0] == cZ
        if inZ:
            outpix = self.outpix[cZ, stroke[:, 1], stroke[:, 2]] > 0
            self.layerz[stroke[~outpix, 1], stroke[~outpix, 2]] = np.array([0, 0, 0, 0])
            cellpix = self.cellpix[cZ, stroke[:, 1], stroke[:, 2]]
            ccol = self.cellcolors.copy()
            if self.selected > 0:
                ccol[self.selected] = np.array([255, 255, 255])
            col2mask = ccol[cellpix]
            if self.masksOn:
                col2mask = np.concatenate(
                    (col2mask, self.opacity * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            else:
                col2mask = np.concatenate(
                    (col2mask, 0 * (cellpix[:, np.newaxis] > 0)), axis=-1
                )
            self.layerz[stroke[:, 1], stroke[:, 2], :] = col2mask
            if self.outlinesOn:
                self.layerz[stroke[outpix, 1], stroke[outpix, 2]] = np.array(
                    self.outcolor
                )
            if delete_points:
                # self.current_point_set = self.current_point_set[:-1*(stroke[:,-1]==1).sum()]
                del self.current_point_set[stroke_ind]
            self.update_layer()

        del self.strokes[stroke_ind]

    def plot_clicked(self, event):
        """
        Handles mouse click events to adjust plot ranges.

            This method responds to mouse click events, specifically left button clicks that are not modified by the Shift or Alt keys.
            When a double-click event is detected, it attempts to set the Y and X ranges of the plot to defined values. If an exception occurs
            during this operation, it falls back to a default Y range.

            Args:
                event: The mouse event that triggered the method.

            Returns:
                None
        """
        if (
            event.button() == QtCore.Qt.LeftButton
            and not event.modifiers()
            & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)
            and not self.removing_region
        ):
            if event.double():
                try:
                    self.p0.setYRange(0, self.Ly + self.pr)
                except:
                    self.p0.setYRange(0, self.Ly)
                self.p0.setXRange(0, self.Lx)

    def cancel_remove_multiple(self):
        """
        Cancels the removal of multiple selected cells.

            This method resets the state of the selected cells by clearing the
            current selection and performing cleanup operations related to the
            removal process.

            Parameters:
                None

            Returns:
                None
        """
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        """
        Clears multi-selected cells by unselecting them.

            This method iterates over a list of previously selected cells and unselects each one.
            After unselecting, it clears the list of selected cells to ensure there are no remaining selections.

            Parameters:
                None

            Returns:
                None
        """
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        """
        Add a region of interest (ROI) to the graphical interface.

            This method adds a specified ROI item to a collection
            and updates the reference for removing that ROI as necessary.

            Args:
                roi: The region of interest item to be added.

            Returns:
                None
        """
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        """
        Remove the specified region of interest (ROI) from the interface.

            This method clears any multi-selected cells, verifies that the provided ROI
            matches the currently active ROI, and then removes it from the graphical interface.
            It also resets the state to indicate that no region is currently being removed.

            Args:
                roi: The region of interest to be removed from the interface.

            Returns:
                None
        """
        self.clear_multi_selected_cells()
        assert roi == self.remove_roi_obj
        self.remove_roi_obj = None
        self.p0.removeItem(roi)
        self.removing_region = False

    def roi_changed(self, roi):
        """
        Updates the selection of cells based on the given region of interest (ROI).

            This method finds cells that overlap with the specified ROI and selects them. It ensures
            that the coordinates of the region are constrained within defined bounds, and it updates
            the selection to reflect which cells are present in the specified area.

            Args:
                roi: The region of interest that defines the area to check for overlapping cells.

            Returns:
                None
        """
        # find the overlapping cells and make them selected
        pos = roi.pos()
        size = roi.size()
        x0 = int(pos.x())
        y0 = int(pos.y())
        x1 = int(pos.x() + size.x())
        y1 = int(pos.y() + size.y())
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > self.Lx:
            x1 = self.Lx
        if y1 > self.Ly:
            y1 = self.Ly

        # find cells in that region
        cell_idxs = np.unique(self.cellpix[self.currentZ, y0:y1, x0:x1])
        cell_idxs = np.trim_zeros(cell_idxs)
        # deselect cells not in region by deselecting all and then selecting the ones in the region
        self.clear_multi_selected_cells()

        for idx in cell_idxs:
            self.select_cell_multi(idx)
            self.removing_cells_list.append(idx)

        self.update_layer()

    def mouse_moved(self, pos):
        """
        Handles mouse movement events and retrieves scene items at the given position.

            This method is called whenever the mouse is moved within the application's window.
            It retrieves the items present in the scene at the specified position.

            Args:
                pos: The position of the mouse pointer in the scene's coordinate space.

            Returns:
                A list of items located at the specified position in the scene.
        """
        items = self.win.scene().items(pos)

    def color_choose(self):
        """
        Selects the current color from the RGB dropdown and updates the plot.

            This method retrieves the selected index from the RGB dropdown menu,
            sets the current view index to zero, and refreshes the plot to reflect
            the changes.

            Parameters:
                None

            Returns:
                None
        """
        self.color = self.RGBDropDown.currentIndex()
        self.view = 0
        self.ViewDropDown.setCurrentIndex(self.view)
        self.update_plot()

    def update_plot(self):
        """
        Update the plot based on the current settings and state.

            This method refreshes the displayed image in the plot based on the currently selected
            view and other parameters, such as whether the image should be resized or if specific
            color channels should be displayed. It adjusts the levels of saturation and handles
            both single-channel and multi-channel images.

            Parameters:
                None

            Returns:
                None
        """
        self.view = self.ViewDropDown.currentIndex()
        self.Ly, self.Lx, _ = self.stack[self.currentZ].shape

        if self.restore and "upsample" in self.restore:
            if self.view != 0:
                if self.view == 3:
                    self.resize = True
                elif len(self.flows[0]) > 0 and self.flows[0].shape[1] == self.Lyr:
                    self.resize = True
                else:
                    self.resize = False
            else:
                self.resize = False
            self.draw_layer()
            self.update_scale()
            self.update_layer()

        if self.view == 0 or self.view == self.ViewDropDown.count() - 1:
            image = (
                self.stack[self.currentZ]
                if self.view == 0
                else self.stack_filtered[self.currentZ]
            )
            if self.nchan == 1:
                # show single channel
                image = image[..., 0]
            if self.color == 0:
                self.img.setImage(image, autoLevels=False, lut=None)
                if self.nchan > 1:
                    levels = np.array(
                        [
                            self.saturation[0][self.currentZ],
                            self.saturation[1][self.currentZ],
                            self.saturation[2][self.currentZ],
                        ]
                    )
                    self.img.setLevels(levels)
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color > 0 and self.color < 4:
                if self.nchan > 1:
                    image = image[:, :, self.color - 1]
                self.img.setImage(image, autoLevels=False, lut=self.cmap[self.color])
                if self.nchan > 1:
                    self.img.setLevels(self.saturation[self.color - 1][self.currentZ])
                else:
                    self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 4:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=None)
                self.img.setLevels(self.saturation[0][self.currentZ])
            elif self.color == 5:
                if self.nchan > 1:
                    image = image.mean(axis=-1)
                self.img.setImage(image, autoLevels=False, lut=self.cmap[0])
                self.img.setLevels(self.saturation[0][self.currentZ])
        else:
            image = np.zeros((self.Ly, self.Lx), np.uint8)
            if len(self.flows) >= self.view - 1 and len(self.flows[self.view - 1]) > 0:
                image = self.flows[self.view - 1][self.currentZ]
            if self.view > 1:
                self.img.setImage(image, autoLevels=False, lut=self.bwr)
            else:
                self.img.setImage(image, autoLevels=False, lut=None)
            self.img.setLevels([0.0, 255.0])

        for r in range(3):
            self.sliders[r].setValue(
                [
                    self.saturation[r][self.currentZ][0],
                    self.saturation[r][self.currentZ][1],
                ]
            )
        self.win.show()
        self.show()

    def update_layer(self):
        """
        Update the layer's image and refresh the display.

            This method updates the layer's image if certain conditions are met (when masks or outlines are enabled)
            and then refreshes the region of interest (ROI) count and the user interface.

            Parameters:
                None

            Returns:
                None
        """
        if self.masksOn or self.outlinesOn:
            # self.draw_layer()
            self.layer.setImage(self.layerz, autoLevels=False)
        self.update_roi_count()
        self.win.show()
        self.show()

    def update_roi_count(self):
        """
        Update the displayed count of Regions of Interest (ROIs).

            This method updates the text of the roi_count attribute to reflect
            the current number of ROIs by setting it to the value of the
            ncells attribute followed by the string " ROIs".

            Parameters:
                None

            Returns:
                None
        """
        self.roi_count.setText(f"{self.ncells} ROIs")

    def add_set(self):
        """
        Add a new set of points to the current configuration if certain conditions are met.

            This method checks if there are any current points in the point set and processes
            strokes accordingly. If the size of the first point set exceeds a specified threshold,
            it adds a mask with the given color and updates the relevant attributes. Otherwise,
            it prints an error message indicating that the cell is too small to draw.

            Returns:
                None: This method does not return any value.

            Raises:
                ValueError: If the current point set is empty or does not meet the required conditions.
        """
        if len(self.current_point_set) > 0:
            while len(self.strokes) > 0:
                self.remove_stroke(delete_points=False)
            if len(self.current_point_set[0]) > 8:
                color = self.colormap[self.ncells, :3]
                median = self.add_mask(points=self.current_point_set, color=color)
                if median is not None:
                    self.removed_cell = []
                    self.toggle_mask_ops()
                    self.cellcolors = np.append(
                        self.cellcolors, color[np.newaxis, :], axis=0
                    )
                    self.ncells += 1
                    self.ismanual = np.append(self.ismanual, True)
                    if self.NZ == 1:
                        # only save after each cell if single image
                        io._save_sets_with_check(self)
            else:
                print("GUI_ERROR: cell too small, not drawn")
            self.current_stroke = []
            self.strokes = []
            self.current_point_set = []
            self.update_layer()

    def add_mask(self, points=None, color=(100, 200, 50), dense=True):
        """
        Adds a mask based on the specified points and optionally draws an outline.

            This method takes a list of stroke points and generates a mask
            by filling in the areas defined by the strokes. It handles overlapping
            regions by reassigning pixels appropriately and computes the median
            of the added stroke coordinates.

            Args:
                points: A list of strokes, where each stroke is an array of
                        coordinates representing drawn points.
                color: A tuple representing the RGB color used for the mask.
                       Defaults to (100, 200, 50).
                dense: A boolean flag indicating whether to use a dense outline
                       method. Defaults to True.

            Returns:
                A list containing the median coordinates of the added mask's
                rows and columns. If the mask cannot be drawn due to insufficient
                pixels, it returns None.
        """
        # points is list of strokes
        points_all = np.concatenate(points, axis=0)

        # loop over z values
        median = []
        zdraw = np.unique(points_all[:, 0])
        z = 0
        ars, acs, vrs, vcs = (
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
            np.zeros(0, "int"),
        )
        for stroke in points:
            stroke = np.concatenate(stroke, axis=0).reshape(-1, 4)
            vr = stroke[:, 1]
            vc = stroke[:, 2]
            # get points inside drawn points
            mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
            pts = np.stack((vc - vc.min() + 2, vr - vr.min() + 2), axis=-1)[
                :, np.newaxis, :
            ]
            mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
            ar, ac = np.nonzero(mask)
            ar, ac = ar + vr.min() - 2, ac + vc.min() - 2
            # get dense outline
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = contours[-2][0][:, 0].T
            vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            # concatenate all points
            ar, ac = np.hstack((np.vstack((vr, vc)), np.vstack((ar, ac))))
            # if these pixels are overlapping with another cell, reassign them
            ioverlap = self.cellpix[z][ar, ac] > 0
            if (~ioverlap).sum() < 10:
                print("GUI_ERROR: cell < 10 pixels without overlaps, not drawn")
                return None
            elif ioverlap.sum() > 0:
                ar, ac = ar[~ioverlap], ac[~ioverlap]
                # compute outline of new mask
                mask = np.zeros((np.ptp(vr) + 4, np.ptp(vc) + 4), np.uint8)
                mask[ar - vr.min() + 2, ac - vc.min() + 2] = 1
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0][:, 0].T
                vr, vc = pvr + vr.min() - 2, pvc + vc.min() - 2
            ars = np.concatenate((ars, ar), axis=0)
            acs = np.concatenate((acs, ac), axis=0)
            vrs = np.concatenate((vrs, vr), axis=0)
            vcs = np.concatenate((vcs, vc), axis=0)

        self.draw_mask(z, ars, acs, vrs, vcs, color)
        median.append(np.array([np.median(ars), np.median(acs)]))

        self.zdraw.append(zdraw)
        d = datetime.datetime.now()
        self.track_changes.append(
            [d.strftime("%m/%d/%Y, %H:%M:%S"), "added mask", [ar, ac]]
        )
        return median

    def draw_mask(self, z, ar, ac, vr, vc, color, idx=None):
        """draw single mask using outlines and area"""
        if idx is None:
            idx = self.ncells + 1
        self.cellpix[z, vr, vc] = idx
        self.cellpix[z, ar, ac] = idx
        self.outpix[z, vr, vc] = idx
        if self.restore and "upsample" in self.restore:
            if self.resize:
                self.cellpix_resize[z, vr, vc] = idx
                self.cellpix_resize[z, ar, ac] = idx
                self.outpix_resize[z, vr, vc] = idx
                self.cellpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
                self.cellpix_orig[
                    z, (ar / self.ratio).astype(int), (ac / self.ratio).astype(int)
                ] = idx
                self.outpix_orig[
                    z, (vr / self.ratio).astype(int), (vc / self.ratio).astype(int)
                ] = idx
            else:
                self.cellpix_orig[z, vr, vc] = idx
                self.cellpix_orig[z, ar, ac] = idx
                self.outpix_orig[z, vr, vc] = idx

                # get upsampled mask
                vrr = (vr.copy() * self.ratio).astype(int)
                vcr = (vc.copy() * self.ratio).astype(int)
                mask = np.zeros((np.ptp(vrr) + 4, np.ptp(vcr) + 4), np.uint8)
                pts = np.stack((vcr - vcr.min() + 2, vrr - vrr.min() + 2), axis=-1)[
                    :, np.newaxis, :
                ]
                mask = cv2.fillPoly(mask, [pts], (255, 0, 0))
                arr, acr = np.nonzero(mask)
                arr, acr = arr + vrr.min() - 2, acr + vcr.min() - 2
                # get dense outline
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = contours[-2][0].squeeze().T
                vrr, vcr = pvr + vrr.min() - 2, pvc + vcr.min() - 2
                # concatenate all points
                arr, acr = np.hstack((np.vstack((vrr, vcr)), np.vstack((arr, acr))))
                self.cellpix_resize[z, vrr, vcr] = idx
                self.cellpix_resize[z, arr, acr] = idx
                self.outpix_resize[z, vrr, vcr] = idx

        if z == self.currentZ:
            self.layerz[ar, ac, :3] = color
            if self.masksOn:
                self.layerz[ar, ac, -1] = self.opacity
            if self.outlinesOn:
                self.layerz[vr, vc] = np.array(self.outcolor)

    def compute_scale(self):
        """
        Computes the scale for a graphical representation based on diameter input.

            This method initializes the diameter, calculates radius padding, and creates
            a color array representation of the radii for use in graphical output. It also
            sets the display range for the plot based on the calculated dimensions.

            Attributes:
                diameter: The diameter value parsed from the UI input.
                pr: The processed radius derived from the diameter.
                radii_padding: The additional padding to be applied to the radius.
                radii: A numpy array representing the radii in RGBA format for visualization.

            Returns:
                None: This method does not return a value but modifies the internal state
                of the object and prepares the graphical representation.
        """
        self.diameter = float(self.Diameter.text())
        self.pr = int(float(self.Diameter.text()))
        self.radii_padding = int(self.pr * 1.25)
        self.radii = np.zeros((self.Ly + self.radii_padding, self.Lx, 4), np.uint8)
        yy, xx = disk(
            [self.Ly + self.radii_padding / 2 - 1, self.pr / 2 + 1],
            self.pr / 2,
            self.Ly + self.radii_padding,
            self.Lx,
        )
        # rgb(150,50,150)
        self.radii[yy, xx, 0] = 150
        self.radii[yy, xx, 1] = 50
        self.radii[yy, xx, 2] = 150
        self.radii[yy, xx, 3] = 255
        self.p0.setYRange(0, self.Ly + self.radii_padding)
        self.p0.setXRange(0, self.Lx)

    def update_scale(self):
        """
        Update the scale image with the current radii and set the levels.

            This method computes the scale, updates the associated image with
            the current radii, sets the image levels for display, and shows
            the updated window.

            Parameters:
                None

            Returns:
                None
        """
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        """
        Redraws the masks on the current layer.

            This method updates the visual representation of the masks by refreshing
            their outlines and drawing them on the specified layer.

            Args:
                outlines: A boolean flag indicating whether to draw the outlines of the masks.
                draw: A boolean flag indicating whether to draw the masks themselves.

            Returns:
                None
        """
        self.draw_layer()

    def draw_masks(self):
        """
        Draws masks on the current layer.

            This method initiates the process of drawing masks by calling the draw_layer method.

            Parameters:
                None

            Returns:
                None
        """
        self.draw_layer()

    def draw_layer(self):
        """
        Draws the current layer based on the specified parameters.

            This method updates the layer representation by considering masks,
            outlines, and restoration options. It sets pixel values based on
            the current layer's configurations including resizing, cell colors,
            and opacity.

            Parameters:
                None

            Returns:
                None
        """
        if self.resize:
            self.Ly, self.Lx = self.Lyr, self.Lxr
        else:
            self.Ly, self.Lx = self.Ly0, self.Lx0

        if self.masksOn or self.outlinesOn:
            if self.restore and "upsample" in self.restore:
                if self.resize:
                    self.cellpix = self.cellpix_resize.copy()
                    self.outpix = self.outpix_resize.copy()
                else:
                    self.cellpix = self.cellpix_orig.copy()
                    self.outpix = self.outpix_orig.copy()

        # print(self.cellpix.shape, self.outpix.shape, self.cellpix.max(), self.outpix.max())
        self.layerz = np.zeros((self.Ly, self.Lx, 4), np.uint8)
        if self.masksOn:
            self.layerz[..., :3] = self.cellcolors[self.cellpix[self.currentZ], :]
            self.layerz[..., 3] = self.opacity * (
                self.cellpix[self.currentZ] > 0
            ).astype(np.uint8)
            if self.selected > 0:
                self.layerz[self.cellpix[self.currentZ] == self.selected] = np.array(
                    [255, 255, 255, self.opacity]
                )
            cZ = self.currentZ
            stroke_z = np.array([s[0][0] for s in self.strokes])
            inZ = np.nonzero(stroke_z == cZ)[0]
            if len(inZ) > 0:
                for i in inZ:
                    stroke = np.array(self.strokes[i])
                    self.layerz[stroke[:, 1], stroke[:, 2]] = np.array(
                        [255, 0, 255, 100]
                    )
        else:
            self.layerz[..., 3] = 0

        if self.outlinesOn:
            self.layerz[self.outpix[self.currentZ] > 0] = np.array(
                self.outcolor
            ).astype(np.uint8)

    def set_restore_button(self):
        """
        Sets the style of denoise buttons based on the current restore settings.

            This method iterates over the denoise buttons and applies different styles
            depending on the keys in the denoise text and the current restore state.
            If a button's corresponding key is not "none" and matches the restore state,
            it is styled as pressed. If the key equals "none" and restore is None,
            the button is also styled as pressed. Otherwise, if the button is enabled,
            it is styled as unpressed.

            Parameters:
                None

            Returns:
                None
        """
        keys = self.denoise_text
        for i, key in enumerate(keys):
            if key != "none" and (self.restore and key in self.restore):
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            elif key == "none" and self.restore is None:
                self.DenoiseButtons[i].setStyleSheet(self.stylePressed)
            else:
                if self.DenoiseButtons[i].isEnabled():
                    self.DenoiseButtons[i].setStyleSheet(self.styleUnpressed)

    def set_normalize_params(self, normalize_params):
        """
        Sets the normalization parameters for the model.

            This method updates the normalization parameters by incorporating default values
            and checking specific parameters for validity. It conditionally applies default
            settings based on the current restoration setting, ensuring that only certain keys
            in the normalization parameters are retained.

            Args:
                normalize_params: A dictionary containing the current normalization parameters
                                  to be updated.

            Returns:
                None: This method modifies the input dictionary in place and does not return
                      any value.
        """
        from cellpose.models import normalize_default

        if self.restore != "filter":
            keys = list(normalize_params.keys()).copy()
            for key in keys:
                if key != "percentile":
                    normalize_params[key] = normalize_default[key]
        normalize_params = {**normalize_default, **normalize_params}
        percentile = self.check_percentile_params(normalize_params["percentile"])
        out = self.check_filter_params(
            normalize_params["sharpen_radius"],
            normalize_params["smooth_radius"],
            normalize_params["tile_norm_blocksize"],
            normalize_params["tile_norm_smooth3D"],
            normalize_params["norm3D"],
            normalize_params["invert"],
        )

    def check_percentile_params(self, percentile):
        """
        Validate and set percentile parameters for normalization.

            This method checks if the provided percentile values are within the valid range
            and whether the upper percentile is greater than the lower percentile. If the
            parameters are invalid or not provided, it sets default values of 1 and 99.
            The normalized percentile values are also updated to reflect the current state.

            Args:
                percentile: A list containing two values representing the lower and upper
                            percentiles to be validated. The method checks that these values
                            are between 0 and 100, and that the upper value is greater than
                            the lower value.

            Returns:
                A list containing the validated lower and upper percentile values.
        """
        # check normalization params
        if percentile is not None and not (
            percentile[0] >= 0
            and percentile[1] > 0
            and percentile[0] < 100
            and percentile[1] <= 100
            and percentile[1] > percentile[0]
        ):
            print(
                "GUI_ERROR: percentiles need be between 0 and 100, and upper > lower, using defaults"
            )
            self.norm_edits[0].setText("1.")
            self.norm_edits[1].setText("99.")
            percentile = [1.0, 99.0]
        elif percentile is None:
            percentile = [1.0, 99.0]
        self.norm_edits[0].setText(str(percentile[0]))
        self.norm_edits[1].setText(str(percentile[1]))
        return percentile

    def check_filter_params(self, sharpen, smooth, tile_norm, smooth3D, norm3D, invert):
        """
        Checks and adjusts filter parameters for image processing.

            This method validates and updates the filter parameters for sharpening,
            smoothing, and normalization, ensuring that they remain within allowable bounds,
            and updates the corresponding user interface elements.

            Args:
                sharpen: The sharpening level to apply. If negative, it is reset to zero.
                smooth: The smoothing level to apply. If negative, it is reset to zero.
                tile_norm: The tile normalization size. If negative, it is reset to zero.
                           If it exceeds the image dimensions, it is disabled.
                smooth3D: The 3D smoothing level to apply. If negative, it is reset to zero.
                norm3D: A flag indicating whether to normalize in 3D.
                invert: A flag indicating whether to invert the image colors.

            Returns:
                A tuple containing the adjusted values of sharpen, smooth, tile_norm,
                smooth3D, norm3D, and invert.
        """
        tile_norm = 0 if tile_norm < 0 else tile_norm
        sharpen = 0 if sharpen < 0 else sharpen
        smooth = 0 if smooth < 0 else smooth
        smooth3D = 0 if smooth3D < 0 else smooth3D
        norm3D = bool(norm3D)
        invert = bool(invert)
        if tile_norm > self.Ly and tile_norm > self.Lx:
            print(
                "GUI_ERROR: tile size (tile_norm) bigger than both image dimensions, disabling"
            )
            tile_norm = 0
        self.filt_edits[0].setText(str(sharpen))
        self.filt_edits[1].setText(str(smooth))
        self.filt_edits[2].setText(str(tile_norm))
        self.filt_edits[3].setText(str(smooth3D))
        self.norm3D_cb.setChecked(norm3D)
        self.invert_cb.setChecked(invert)
        return sharpen, smooth, tile_norm, smooth3D, norm3D, invert

    def get_normalize_params(self):
        """
        Retrieve normalization parameters based on user input.

            This method gathers normalization parameters from various UI components,
            validates the provided values, and constructs a dictionary of these parameters
            that will be used for data normalization.

            Returns:
                A dictionary containing the normalization parameters including
                percentile values, 3D normalization flag, and optional filter parameters
                if applicable.

            Raises:
                ValueError: If any input values are invalid during parameter checks.
        """
        percentile = [
            float(self.norm_edits[0].text()),
            float(self.norm_edits[1].text()),
        ]
        self.check_percentile_params(percentile)
        normalize_params = {"percentile": percentile}
        norm3D = self.norm3D_cb.isChecked()
        normalize_params["norm3D"] = norm3D
        if self.restore == "filter":
            sharpen = float(self.filt_edits[0].text())
            smooth = float(self.filt_edits[1].text())
            tile_norm = float(self.filt_edits[2].text())
            smooth3D = float(self.filt_edits[3].text())
            invert = self.invert_cb.isChecked()
            out = self.check_filter_params(
                sharpen, smooth, tile_norm, smooth3D, norm3D, invert
            )
            sharpen, smooth, tile_norm, smooth3D, norm3D, invert = out
            normalize_params["sharpen_radius"] = sharpen
            normalize_params["smooth_radius"] = smooth
            normalize_params["tile_norm_blocksize"] = tile_norm
            normalize_params["tile_norm_smooth3D"] = smooth3D
            normalize_params["invert"] = invert

        from cellpose.models import normalize_default

        normalize_params = {**normalize_default, **normalize_params}

        return normalize_params

    def compute_saturation(self, return_img=False):
        """
        Compute the saturation values for an image stack.

            This method normalizes the image stack by applying optional smoothing, sharpening, and tile normalization
            processes. It calculates saturation levels based on specified percentiles and stores these values for
            each channel in the image. The method also handles grayscale images appropriately.

            Args:
                return_img: A flag that indicates whether to return the normalized image.

            Returns:
                The method returns a list of saturation values corresponding to each channel in the image.
        """
        norm = self.get_normalize_params()
        print(norm)
        sharpen, smooth = norm["sharpen_radius"], norm["smooth_radius"]
        percentile = norm["percentile"]
        tile_norm = norm["tile_norm_blocksize"]
        invert = norm["invert"]
        norm3D = norm["norm3D"]
        smooth3D = norm["tile_norm_smooth3D"]
        tile_norm = norm["tile_norm_blocksize"]

        # if grayscale, use gray img
        channels = self.get_channels()
        if channels[0] == 0:
            img_norm = self.stack.mean(axis=-1, keepdims=True)
        elif sharpen > 0 or smooth > 0 or tile_norm > 0:
            img_norm = self.stack.copy()
        else:
            img_norm = self.stack

        if sharpen > 0 or smooth > 0 or tile_norm > 0:
            self.clear_restore()
            self.restore = "filter"
            print(
                "GUI_INFO: computing filtered image because sharpen > 0 or tile_norm > 0"
            )
            print(
                "GUI_WARNING: will use memory to create filtered image -- make sure to have RAM for this"
            )
            img_norm = self.stack.copy()
            if sharpen > 0 or smooth > 0:
                img_norm = smooth_sharpen_img(
                    self.stack, sharpen_radius=sharpen, smooth_radius=smooth
                )

            if tile_norm > 0:
                img_norm = normalize99_tile(
                    img_norm,
                    blocksize=tile_norm,
                    lower=percentile[0],
                    upper=percentile[1],
                    smooth3D=smooth3D,
                    norm3D=norm3D,
                )
            # convert to 0->255
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= img_norm_max - img_norm_min
            img_norm *= 255
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)
        elif invert:
            img_norm = self.stack.copy()
        else:
            img_norm = (
                self.stack
                if self.restore is None or self.restore == "filter"
                else self.stack_filtered
            )

        self.saturation = []
        for c in range(img_norm.shape[-1]):
            self.saturation.append([])
            if np.ptp(img_norm[..., c]) > 1e-3:
                if norm3D:
                    x01 = np.percentile(img_norm[..., c], percentile[0])
                    x99 = np.percentile(img_norm[..., c], percentile[1])
                    if invert:
                        x01i = 255.0 - x99
                        x99i = 255.0 - x01
                        x01, x99 = x01i, x99i
                    for n in range(self.NZ):
                        self.saturation[-1].append([x01, x99])
                else:
                    for z in range(self.NZ):
                        if self.NZ > 1:
                            x01 = np.percentile(img_norm[z, :, :, c], percentile[0])
                            x99 = np.percentile(img_norm[z, :, :, c], percentile[1])
                        else:
                            x01 = np.percentile(img_norm[..., c], percentile[0])
                            x99 = np.percentile(img_norm[..., c], percentile[1])
                        if invert:
                            x01i = 255.0 - x99
                            x99i = 255.0 - x01
                            x01, x99 = x01i, x99i
                        self.saturation[-1].append([x01, x99])
            else:
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.0])
        # if only 2 restore channels, add blue
        if len(self.saturation) < 3:
            for i in range(3 - len(self.saturation)):
                self.saturation.append([])
                for n in range(self.NZ):
                    self.saturation[-1].append([0, 255.0])
        print(self.saturation[2][self.currentZ])

        if invert:
            img_norm = 255.0 - img_norm
            self.stack_filtered = img_norm
            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

        if img_norm.shape[-1] == 1:
            self.saturation.append(self.saturation[0])
            self.saturation.append(self.saturation[0])

        self.autobtn.setChecked(True)
        self.update_plot()

    def chanchoose(self, image):
        """
        Selects and returns specified channels from the input image.

            This method checks the dimensions of the provided image and the channel selection options.
            If the image has more than two dimensions and multiple channels are available, it
            allows the user to choose which channels to return. If no specific channels are chosen,
            it returns the average of the channels.

            Returns:
                The selected channels from the image. If only one channel is selected, it returns
                the mean across channels as a 3D array. If multiple channels are selected, it returns
                the respective channels as a 3D sub-array.

        """
        if image.ndim > 2 and self.nchan > 1:
            if self.ChannelChoose[0].currentIndex() == 0:
                return image.mean(axis=-1, keepdims=True)
            else:
                chanid = [self.ChannelChoose[0].currentIndex() - 1]
                if self.ChannelChoose[1].currentIndex() > 0:
                    chanid.append(self.ChannelChoose[1].currentIndex() - 1)
                return image[:, :, chanid]
        else:
            return image

    def get_model_path(self, custom=False):
        """
        Retrieve the file path of the selected model.

            This method determines the file path based on whether a custom model
            is specified. If a custom model is chosen, it uses the current selection
            from the GUI dropdown. Otherwise, it retrieves the path based on predefined
            network names.

            Args:
                custom: A boolean indicating whether to use a custom model.

            Returns:
                A string representing the file path of the selected model.
        """
        if custom:
            self.current_model = self.ModelChooseC.currentText()
            self.current_model_path = os.fspath(
                models.MODEL_DIR.joinpath(self.current_model)
            )
        else:
            self.current_model = self.net_names[
                max(0, self.ModelChooseB.currentIndex() - 1)
            ]
            self.current_model_path = models.model_path(self.current_model)

    def initialize_model(self, model_name=None, custom=False):
        """
        Initializes the model based on the specified parameters.

            This method sets up the Cellpose model based on the provided model name or custom settings.
            It checks for valid model names, retrieves the appropriate model path, and handles GPU settings.
            If the specified model is not found, it raises a ValueError.

            Args:
                model_name: The name of the model to initialize. If not provided, a custom model can be specified.
                del_name: An optional parameter that may be used to specify a model to delete (not implemented in this snippet).
                custom: A boolean indicating whether a custom model should be used instead of a predefined one.

            Returns:
                None: This method does not return a value but initializes the model attribute within the class.
        """
        if model_name == "dataset-specific models":
            raise ValueError("need to specify model (use dropdown)")
        elif model_name is None or custom:
            self.get_model_path(custom=custom)
            if not os.path.exists(self.current_model_path):
                raise ValueError("need to specify model (use dropdown)")

        if model_name is None or not isinstance(model_name, str):
            self.model = models.CellposeModel(
                gpu=self.useGPU.isChecked(), pretrained_model=self.current_model_path
            )
        else:
            self.current_model = model_name
            if self.current_model == "cyto" or self.current_model == "nuclei":
                self.current_model_path = models.model_path(self.current_model, 0)
            else:
                self.current_model_path = os.fspath(
                    models.MODEL_DIR.joinpath(self.current_model)
                )

            if self.current_model != "cyto3":
                diam_mean = 17.0 if self.current_model == "nuclei" else 30.0
                self.model = models.CellposeModel(
                    gpu=self.useGPU.isChecked(),
                    diam_mean=diam_mean,
                    model_type=self.current_model,
                )
            else:
                self.model = models.Cellpose(
                    gpu=self.useGPU.isChecked(), model_type=self.current_model
                )

    def add_model(self):
        """
        Add a model to the system.

            This method integrates a model into the array of available models
            for the system by calling an internal method.

            Parameters:
                None

            Returns:
                None
        """
        io._add_model(self)
        return

    def remove_model(self):
        """
        Removes the current model.

            This method calls an internal function to remove the current model from the system.

            Parameters:
                None

            Returns:
                None
        """
        io._remove_model(self)
        return

    def new_model(self):
        """
        Creates and trains a new model.

            This method initiates the process of training a new model. It checks if the data is in a two-dimensional format,
            collects the required training data, and launches a training window for user interaction. If training is confirmed
            by the user, the model is trained with the provided datasets.

            Raises:
                ValueError: If the data is in a three-dimensional format.

            Returns:
                None: This method does not return a value.
        """
        if self.NZ != 1:
            print("ERROR: cannot train model on 3D data")
            return

        # train model
        image_names = self.get_files()[0]
        (
            self.train_data,
            self.train_labels,
            self.train_files,
            restore,
            normalize_params,
        ) = io._get_train_set(image_names)
        TW = guiparts.TrainWindow(self, models.MODEL_NAMES)
        train = TW.exec_()
        if train:
            self.logger.info(
                f"training with {[os.path.split(f)[1] for f in self.train_files]}"
            )
            self.train_model(restore=restore, normalize_params=normalize_params)
        else:
            print("GUI_INFO: training cancelled")

    def train_model(self, restore=None, normalize_params=None):
        """
        Train a new segmentation model using the specified training parameters.

            This method initializes and trains a Cellpose model for image segmentation based on
            the provided training data and parameters. It also logs the progress, manages the GUI
            state, and saves the model and training losses.

            Args:
                restore: The state of the model to restore (not detailed).
                normalize_params: Parameters for normalization of the training data (optional).

            Returns:
                None: This method does not return any value, but it updates the state of the model
                and saves necessary outputs.
        """
        from cellpose.models import normalize_default

        if normalize_params is None:
            normalize_params = copy.deepcopy(normalize_default)
        if self.training_params["model_index"] < len(models.MODEL_NAMES):
            model_type = models.MODEL_NAMES[self.training_params["model_index"]]
            self.logger.info(f"training new model starting at model {model_type}")
        else:
            model_type = None
            self.logger.info(f"training new model starting from scratch")
        self.current_model = model_type
        self.channels = self.training_params["channels"]

        self.logger.info(
            f"training with chan = {self.ChannelChoose[0].currentText()}, chan2 = {self.ChannelChoose[1].currentText()}"
        )

        self.model = models.CellposeModel(
            gpu=self.useGPU.isChecked(), model_type=model_type
        )
        self.SizeButton.setEnabled(False)
        save_path = os.path.dirname(self.filename)

        print("GUI_INFO: name of new model: " + self.training_params["model_name"])
        print(f"GUI_INFO: SGD activated: {self.training_params['SGD']}")
        self.new_model_path, train_losses = train.train_seg(
            self.model.net,
            train_data=self.train_data,
            train_labels=self.train_labels,
            channels=self.channels,
            normalize=normalize_params,
            min_train_masks=0,
            save_path=save_path,
            nimg_per_epoch=max(8, len(self.train_data)),
            learning_rate=self.training_params["learning_rate"],
            weight_decay=self.training_params["weight_decay"],
            n_epochs=self.training_params["n_epochs"],
            SGD=self.training_params["SGD"],
            model_name=self.training_params["model_name"],
        )[:2]
        # save train losses
        np.save(str(self.new_model_path) + "_train_losses.npy", train_losses)
        # run model on next image
        io._add_model(self, self.new_model_path)
        diam_labels = self.model.net.diam_labels.item()  # .copy()
        self.new_model_ind = len(self.model_strings)
        self.autorun = True
        channels = self.channels.copy()
        self.clear_all()
        # keep same channels
        self.ChannelChoose[0].setCurrentIndex(channels[0])
        self.ChannelChoose[1].setCurrentIndex(channels[1])
        self.diameter = diam_labels
        self.Diameter.setText("%0.2f" % self.diameter)
        self.logger.info(f">>>> diameter set to diam_labels ( = {diam_labels: 0.3f} )")
        self.restore = restore
        self.set_normalize_params(normalize_params)
        self.get_next_image(load_seg=False)

        self.compute_segmentation(custom=True)
        self.logger.info(
            f"!!! computed masks for {os.path.split(self.filename)[1]} from new model !!!"
        )

    def compute_restore(self):
        """
        Computes the restoration of an image based on the specified restoration method.

            This method checks if a restoration process is required and executes
            the corresponding image restoration operations, such as denoising or
            saturation adjustment, based on the defined restore criteria. Depending
            on the type of restoration specified, it may adjust the current index
            of a denoising selection component and set the appropriate diameter
            value according to the restoration parameters.

            Parameters:
                None

            Returns:
                None
        """
        if self.restore:
            self.logger.info(f"running image restoration {self.restore}")
            if self.restore != "filter":
                rstr = self.restore.split("_")
                model_type = rstr[0]
                if len(rstr) > 1:
                    dset = rstr[1]
                    if dset == "cyto3":
                        self.DenoiseChoose.setCurrentIndex(0)
                    else:
                        self.DenoiseChoose.setCurrentIndex(1)
                if "upsample" in self.restore:
                    i = self.DenoiseChoose.currentIndex()
                    diam_up = 30.0 if i == 0 or i == 1 else 17.0
                    print(diam_up, self.ratio)
                    self.Diameter.setText(str(diam_up / self.ratio))
                self.compute_denoise_model(model_type=model_type)
            else:
                self.compute_saturation()

    def get_thresholds(self):
        """
        Retrieve the flow and cell probability thresholds.

            This method attempts to read the flow and cell probability thresholds from
            the corresponding text fields, converting them to float values. If the
            flow threshold is zero or the NZ attribute is greater than one, the flow
            threshold is set to None. In case of any exceptions during this process,
            default values are set, and an appropriate message is printed.

            Returns:
                A tuple containing the flow threshold and the cell probability threshold.
                If the flow threshold is invalid, the first element of the tuple will be
                None (or 0.4 by default), and the second element will reflect the valid
                cell probability threshold (0.0 by default if invalid).
        """
        try:
            flow_threshold = float(self.flow_threshold.text())
            cellprob_threshold = float(self.cellprob_threshold.text())
            if flow_threshold == 0.0 or self.NZ > 1:
                flow_threshold = None
            return flow_threshold, cellprob_threshold
        except Exception as e:
            print(
                "flow threshold or cellprob threshold not a valid number, setting to defaults"
            )
            self.flow_threshold.setText("0.4")
            self.cellprob_threshold.setText("0.0")
            return 0.4, 0.0

    def compute_cprob(self):
        """
        Computes the probability masks based on flow data and specified thresholds.

            This method checks if the recomputation of masks is necessary, retrieves
            flow and cell probability thresholds, and computes the masks accordingly.
            The resulting masks are then made available for the GUI representation.

            Parameters:
                None

            Returns:
                None
        """
        if self.recompute_masks:
            flow_threshold, cellprob_threshold = self.get_thresholds()
            if flow_threshold is None:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, no flow error threshold"
                    % (cellprob_threshold)
                )
            else:
                self.logger.info(
                    "computing masks with cell prob=%0.3f, flow error threshold=%0.3f"
                    % (cellprob_threshold, flow_threshold)
                )
            maski = dynamics.resize_and_compute_masks(
                self.flows[4][:-1],
                self.flows[4][-1],
                p=self.flows[3].copy(),
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                resize=self.cellpix.shape[-2:],
            )[0]

            self.masksOn = True
            if not self.OCheckBox.isChecked():
                self.MCheckBox.setChecked(True)
            if maski.ndim < 3:
                maski = maski[np.newaxis, ...]
            self.logger.info("%d cells found" % (len(np.unique(maski)[1:])))
            io._masks_to_gui(self, maski, outlines=None)
            self.show()

    def compute_denoise_model(self, model_type=None):
        """
        Computes the denoised model based on the selected parameters and model type.

            This method initializes a denoising model, processes input image data,
            applies normalization, and manages the upsampling of images if necessary.
            It also updates the status of the GUI and calculates saturation levels for
            the processed images.

            Args:
                model_type: A string representing the type of model to be used for denoising.

            Returns:
                None: The method does not return a value but updates internal state and the GUI components based on processing results.
        """
        self.progress.setValue(0)
        try:
            tic = time.time()
            nstr = self.DenoiseChoose.currentText()
            nstr.replace("-", "")
            self.clear_restore()
            model_name = model_type + "_" + nstr
            print(model_name)
            # denoising model
            self.denoise_model = denoise.DenoiseModel(
                gpu=self.useGPU.isChecked(), model_type=model_name
            )
            self.progress.setValue(10)
            diam_up = 30.0 if "cyto" in model_name else 17.0

            # params
            channels = self.get_channels()
            self.diameter = float(self.Diameter.text())
            normalize_params = self.get_normalize_params()
            print("GUI_INFO: channels: ", channels)
            print("GUI_INFO: normalize_params: ", normalize_params)
            print("GUI_INFO: diameter (before upsampling): ", self.diameter)

            data = self.stack.copy()
            print(data.shape)
            self.Ly, self.Lx = data.shape[-3:-1]
            if "upsample" in model_name:
                # get upsampling factor
                if self.diameter >= diam_up:
                    print(
                        f"GUI_ERROR: cannot upsample, already set to pixel diameter >= {diam_up}"
                    )
                    self.progress.setValue(0)
                    return
                self.ratio = diam_up / self.diameter
                print(
                    "GUI_WARNING: upsampling image, this will also duplicate mask layer and resize it, will use more RAM"
                )
                print(
                    f"GUI_INFO: upsampling image to {diam_up} pixel diameter ({self.ratio:0.2f} times)"
                )
                self.Lyr, self.Lxr = int(self.Ly * self.ratio), int(
                    self.Lx * self.ratio
                )
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                # moved resize into eval
                # data = resize_image(data, Ly=self.Lyr, Lx=self.Lxr)
                # self.diameter = diam_up
                # self.Diameter.setText(str(diam_up))
            else:
                self.Lyr, self.Lxr = self.Ly, self.Lx
                self.Ly0, self.Lx0 = self.Ly, self.Lx
                diam_up = self.diameter

            img_norm = self.denoise_model.eval(
                data,
                channels=channels,
                z_axis=0,
                channel_axis=3,
                diameter=self.diameter,
                normalize=normalize_params,
            )
            print(img_norm.shape)
            self.diameter = diam_up
            self.Diameter.setText(str(diam_up))

            if img_norm.ndim == 2:
                img_norm = img_norm[:, :, np.newaxis]
            if img_norm.ndim == 3:
                img_norm = img_norm[np.newaxis, ...]

            self.progress.setValue(100)
            self.logger.info(
                f"{model_name} finished in %0.3f sec" % (time.time() - tic)
            )

            # compute saturation
            percentile = normalize_params["percentile"]
            img_norm_min = img_norm.min()
            img_norm_max = img_norm.max()
            chan = [0] if channels[0] == 0 else [channels[0] - 1, channels[1] - 1]
            self.saturation = [[], [], []]
            for c in range(img_norm.shape[-1]):
                if np.ptp(img_norm[..., c]) > 1e-3:
                    img_norm[..., c] -= img_norm_min
                    img_norm[..., c] /= img_norm_max - img_norm_min
                for z in range(self.NZ):
                    x01 = np.percentile(img_norm[z, :, :, c], percentile[0]) * 255.0
                    x99 = np.percentile(img_norm[z, :, :, c], percentile[1]) * 255.0
                    self.saturation[chan[c]].append([x01, x99])
            notchan = np.ones(3, "bool")
            notchan[np.array(chan)] = False
            notchan = np.nonzero(notchan)[0]
            for c in notchan:
                for z in range(self.NZ):
                    self.saturation[c].append([0, 255.0])

            img_norm *= 255.0
            self.autobtn.setChecked(True)

            # assign to denoised channels
            self.stack_filtered = np.zeros(
                (self.NZ, self.Lyr, self.Lxr, self.stack.shape[-1]), "float32"
            )
            for i, c in enumerate(chan[: img_norm.shape[-1]]):
                for z in range(self.NZ):
                    self.stack_filtered[z, :, :, c] = img_norm[z, :, :, i]

            # make upsampled masks
            if model_type == "upsample":
                self.cellpix_orig = self.cellpix.copy()
                self.outpix_orig = self.outpix.copy()
                self.cellpix_resize = cv2.resize(
                    self.cellpix_orig[0],
                    (self.Lxr, self.Lyr),
                    interpolation=cv2.INTER_NEAREST,
                )[np.newaxis, :, :]
                outlines = masks_to_outlines(self.cellpix_resize[0])[np.newaxis, :, :]
                self.outpix_resize = outlines * self.cellpix_resize

            self.restore = model_name

            # draw plot
            if model_type == "upsample":
                self.resize = True
            else:
                self.resize = False
            self.draw_layer()
            self.update_layer()
            self.update_scale()
            # if denoised in grayscale, show in grayscale
            if channels[0] == 0:
                self.RGBDropDown.setCurrentIndex(4)

            self.ViewDropDown.model().item(self.ViewDropDown.count() - 1).setEnabled(
                True
            )
            self.ViewDropDown.setCurrentIndex(self.ViewDropDown.count() - 1)

            self.update_plot()

        except Exception as e:
            print("ERROR: %s" % e)

    def compute_segmentation(self, custom=False, model_name=None, load_model=True):
        """
        Computes the segmentation of a given data stack using a deep learning model.

        This method initializes the model (if specified), processes input data, and computes
        segmentation masks and flow data based on various parameters. It updates the progress
        of the computation and handles potential errors gracefully.

        Args:
            custom: A flag indicating whether to use a custom model.
            model_name: The name of the model to load.
            load_model: A flag indicating whether to load the model before computation.

        Returns:
            None: The method primarily updates the object's state, including storing
            computed segmentation results and progress status.
        """
        self.progress.setValue(0)
        try:
            tic = time.time()
            self.clear_all()
            self.flows = [[], [], []]
            if load_model:
                self.initialize_model(model_name=model_name, custom=custom)
            self.progress.setValue(10)
            do_3D = self.load_3D
            stitch_threshold = (
                float(self.stitch_threshold.text())
                if not isinstance(self.stitch_threshold, float)
                else self.stitch_threshold
            )
            anisotropy = (
                float(self.anisotropy.text())
                if not isinstance(self.anisotropy, float)
                else self.anisotropy
            )
            flow3D_smooth = (
                float(self.flow3D_smooth.text())
                if not isinstance(self.flow3D_smooth, float)
                else self.flow3D_smooth
            )
            min_size = (
                int(self.min_size.text())
                if not isinstance(self.min_size, int)
                else self.min_size
            )
            resample = (
                self.resample.isChecked()
                if not isinstance(self.resample, bool)
                else self.resample
            )

            do_3D = False if stitch_threshold > 0.0 else do_3D

            channels = self.get_channels()
            if self.restore is not None and self.restore != "filter":
                data = self.stack_filtered.copy().squeeze()
            else:
                data = self.stack.copy().squeeze()
            flow_threshold, cellprob_threshold = self.get_thresholds()
            self.diameter = float(self.Diameter.text())
            niter = max(0, int(self.niter.text()))
            niter = None if niter == 0 else niter
            normalize_params = self.get_normalize_params()
            print(normalize_params)
            try:
                masks, flows = self.model.eval(
                    data,
                    channels=channels,
                    diameter=self.diameter,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    do_3D=do_3D,
                    niter=niter,
                    normalize=normalize_params,
                    stitch_threshold=stitch_threshold,
                    anisotropy=anisotropy,
                    resample=resample,
                    flow3D_smooth=flow3D_smooth,
                    min_size=min_size,
                    progress=self.progress,
                    z_axis=0 if self.NZ > 1 else None,
                )[:2]
            except Exception as e:
                print("NET ERROR: %s" % e)
                self.progress.setValue(0)
                return

            self.progress.setValue(75)

            # convert flows to uint8 and resize to original image size
            flows_new = []
            flows_new.append(flows[0].copy())  # RGB flow
            flows_new.append(
                (np.clip(normalize99(flows[2].copy()), 0, 1) * 255).astype("uint8")
            )  # cellprob
            if self.load_3D:
                if stitch_threshold == 0.0:
                    flows_new.append((flows[1][0] / 10 * 127 + 127).astype("uint8"))
                else:
                    flows_new.append(np.zeros(flows[1][0].shape, dtype="uint8"))

            if not self.load_3D:
                if self.restore and "upsample" in self.restore:
                    self.Ly, self.Lx = self.Lyr, self.Lxr

                if flows_new[0].shape[-3:-1] != (self.Ly, self.Lx):
                    self.flows = []
                    for j in range(len(flows_new)):
                        self.flows.append(
                            resize_image(
                                flows_new[j],
                                Ly=self.Ly,
                                Lx=self.Lx,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                else:
                    self.flows = flows_new
            else:
                if not resample:
                    self.flows = []
                    Lz, Ly, Lx = self.NZ, self.Ly, self.Lx
                    Lz0, Ly0, Lx0 = flows_new[0].shape[:3]
                    print("GUI_INFO: resizing flows to original image size")
                    for j in range(len(flows_new)):
                        flow0 = flows_new[j]
                        if Ly0 != Ly:
                            flow0 = resize_image(
                                flow0,
                                Ly=Ly,
                                Lx=Lx,
                                no_channels=flow0.ndim == 3,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        if Lz0 != Lz:
                            flow0 = np.swapaxes(
                                resize_image(
                                    np.swapaxes(flow0, 0, 1),
                                    Ly=Lz,
                                    Lx=Lx,
                                    no_channels=flow0.ndim == 3,
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                0,
                                1,
                            )
                        self.flows.append(flow0)
                else:
                    self.flows = flows_new

            # add first axis
            if self.NZ == 1:
                masks = masks[np.newaxis, ...]
                self.flows = [
                    self.flows[n][np.newaxis, ...] for n in range(len(self.flows))
                ]

            self.logger.info(
                "%d cells found with model in %0.3f sec"
                % (len(np.unique(masks)[1:]), time.time() - tic)
            )
            self.progress.setValue(80)
            z = 0

            io._masks_to_gui(self, masks, outlines=None)
            self.masksOn = True
            self.MCheckBox.setChecked(True)
            self.keepMask.setEnabled(True)
            self.saveMasks.setEnabled(True)

            self.progress.setValue(100)
            if self.restore != "filter" and self.restore is not None:
                self.compute_saturation()
            if not do_3D and not stitch_threshold > 0:
                self.recompute_masks = True
            else:
                self.recompute_masks = False
        except Exception as e:
            print("ERROR: %s" % e)
