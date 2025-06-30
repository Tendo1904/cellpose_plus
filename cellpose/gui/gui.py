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

    This class represents a slider widget that allows users to select a value from a specified range. It includes functionality for updating the slider's value based on user interaction and notifying the parent object of changes.

    Attributes:
    - value: The current value of the slider.
    - min_value: The minimum value the slider can represent.
    - max_value: The maximum value the slider can represent.
    - step: The increment value by which the slider moves.

    Methods:
    - __init__
    - set_value
    - get_value
    - on_value_changed

    The `__init__` method initializes the slider's properties such as its range and initial value. The `set_value` method allows setting the current slider value programmatically, while `get_value` retrieves the current value. The `on_value_changed` method is called to handle events when the value of the slider changes, updating the parent component accordingly.
    """

    def __init__(self, parent, name, color):
        """
        Initializes the MainW class, configuring the main application window and setting up UI components.

        Args:
            image: Optional parameter that can be used to load an image at startup.
            logger: Optional logger instance for logging purposes.

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
        No valid docstring found.
        """
        parent.level_change(self.name)


class QHLine(QFrame):
    """
    QHLine is a class that represents a horizontal line within a graphical user interface, allowing customization of its appearance.

    Attributes:
    - frame_shape: Represents the shape of the frame, which is set to a horizontal line.
    - line_width: Defines the width of the line for visual presentation.

    Methods:
    - __init__:
    """

    def __init__(self):
        """
        Initializes a QHLine object by setting its frame shape to a horizontal line and configuring its line width.

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
    No valid docstring found.
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
    Creates a spectral colormap using predefined RGB values.

    This method constructs a color map by defining the red, green, and blue components
    for a range of colors. It utilizes numpy arrays to specify the intensity of each color channel
    at different positions, resulting in a smooth transition across the spectrum.

    Returns:
        A color map object that represents the spectral colormap.
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
    Creates a single channel colormap using a specified channel index.

    Args:
        cm: The index of the color channel to be used, which determines the color gradient (0 for red, 1 for green, 2 for blue).

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
    Runs the Cellpose GUI application, initializing the environment and downloading necessary resources.

    Args:
        image: Optional parameter for the image to be processed by the application.

    Returns:
        An integer indicating the exit status of the application.
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
    MainW class manages the main application window and user interface for an image processing tool, providing functionalities for image loading, editing, and segmentation.

    Attributes:
    - autosave: Indicates whether autosave is enabled or not.

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
    - delete_restore
    - clear_restore
    - brush_choose
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
    """

    def __init__(self, image=None, logger=None):
        """
        Initializes a new instance of the class and sets up the slider properties.

        This constructor initializes the slider with a horizontal orientation, disables
        the slider, and connects the value change event to a method in the parent object.
        It also sets the slider's style to have a transparent background.

        Args:
            parent: The parent object to which the slider's value change event is connected.
            name: A string representing the name of the slider.
            color: The color of the slider (though it is not used in the current implementation).

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
        Displays a help window to provide assistance to the user.

        This method creates an instance of the HelpWindow class and presents it to the user.

        Returns:
            None: This method does not return a value.
        """
        HW = guiparts.HelpWindow(self)
        HW.show()

    def train_help_window(self):
        """
        No valid docstring found.
        """
        THW = guiparts.TrainHelpWindow(self)
        THW.show()

    def gui_window(self):
        """
        Displays a graphical user interface window using the ExampleGUI class.

        This method initializes an instance of the ExampleGUI class and shows the GUI window.

        Parameters:
        None

        Returns:
        None
        """
        EG = guiparts.ExampleGUI(self)
        EG.show()

    def make_buttons(self):
        """
        Creates and configures various UI buttons and controls for the application.

        This method constructs a user interface layout that includes buttons, combo boxes, checkboxes,
        and sliders necessary for interacting with the application. The UI elements created pertain
        to features such as adjusting visibility, color settings, drawing tools, and segmentation options.

        Returns:
            int: The number of rows added to the layout after inserting all the buttons and controls.
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
        Updates the pixel-to-millimeter conversion factor using the value from the pixTomicro input field.

        Parameters:
        self: Refers to the instance of the class where this method is defined.

        Returns:
        None
        """
        self.px_to_mm = float(self.pixTomicro.text())

    def level_change(self, r):
        """
        Changes the saturation levels for a specified color based on the slider value.

        This method updates the saturation values for the specified color channel and the current Z level. If automation is not enabled, it replicates the slider value across all saturation levels for that color channel. Finally, it refreshes the plot to reflect the changes.

        Parameters:
        - r: The index of the color channel (0 for red, 1 for green, 2 for blue).
        - self.loaded: A boolean indicating whether the saturation data is loaded.
        - self.sliders: A list containing the slider objects for each color channel.
        - self.saturation: A 2D list representing saturation values for each color channel across multiple Z levels.
        - self.currentZ: The current Z level index.
        - self.autobtn: A button object that indicates whether automation is enabled.

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
        Handles key press events and performs corresponding actions based on the key pressed.

        This method listens for key press events, checks the current state of the application,
        and updates the user interface accordingly. It supports various shortcuts for functionality
        such as navigating images, toggling options, changing colors, and modifying brush selections.

        Args:
            event: The key press event containing information about the key that was pressed.

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
        Sets the autosave state based on the status of a checkbox.

        This method checks if a specific checkbox (SCheckBox) is checked.
        If it is checked, autosave is enabled; otherwise, it is disabled.

        Attributes:
            autosave: A boolean indicating whether autosave is enabled or not.

        Returns:
            None
        """
        if self.SCheckBox.isChecked():
            self.autosave = True
        else:
            self.autosave = False

    def check_gpu(self, torch=True):
        """
        Checks if a GPU is available for use with Torch and updates the user interface accordingly.

        This method disables the GPU checkbox by default and checks if a GPU is available. If a GPU is detected, it enables and checks the GPU option in the user interface. If no GPU is found, it changes the color of the GPU option to indicate its unavailability.

        Parameters:
        - None

        Returns:
        - None
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
        Returns the selected channels from the user interface and adjusts them based on specific criteria.

        This method retrieves the indices of the currently selected channels from the user interface.
        It applies certain conditions based on the current model and the number of channels available,
        ensuring that the selection adheres to the restrictions of the underlying data.

        Returns:
            A list of integers representing the adjusted channel indices.
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
        Chooses and initializes a model based on the user's selection from a GUI dropdown.

        This method retrieves the currently selected model from either a custom model selection dropdown
        or a default model selection dropdown. It then initializes the specified model and updates the
        diameter display accordingly.

        Parameters:
        custom - A boolean indicating whether to use the custom model dropdown. If False, the default model dropdown is used.
        m - A boolean variable that is not utilized within the method.

        Returns:
        None - This method does not return any value.
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
        Calibrate the size of cells using a specified model.

        This method initializes the model, evaluates the size of cells in the current
        image stack, and updates the displayed diameter and scale accordingly. It ensures
        the estimated diameters are not less than a minimum threshold and logs the
        calculated values.

        Parameters:
        - model_name: The name of the model used for calibration.
        - stack: The collection of images to be analyzed.
        - currentZ: The index of the current image slice.
        - progress: The progress bar to update the calibration status.
        - logger: The logger instance to log the estimated diameters.
        - Diameter: UI component to display the estimated diameter.
        - diameter: A variable to store the estimated diameter.

        Returns:
        - None: This method does not return a value, but updates UI components and
          state variables.
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
        Toggles the visibility of the scale item in the plot.

        This method adds or removes the scale item from the plot based on the
        current state. If the scale is currently visible, it will be removed;
        if it is not visible, it will be added.

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
        Enables various buttons and UI components in the application.

        This method is responsible for enabling specific buttons associated with
        the model, style, denoising operations, and sliders based on the current
        state of the application. It also handles certain conditions, such as
        disabling specific buttons when 3D loading is active.

        Parameters:
        - self.model_strings: The list determining the state of model-related buttons.
        - self.StyleButtons: A list of style buttons to be enabled or disabled.
        - self.DenoiseButtons: A list of denoise buttons that may be adjusted.
        - self.load_3D: A boolean indicating if 3D loading is active.
        - self.sliders: A list of sliders that will be enabled based on the channel.
        - self.nchan: The number of channels which affects the enabling of sliders.
        - self.filename: The name of the file to be displayed in the window title.

        Returns:
        None: This method does not return any value, but modifies the UI state.
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
        Disables various buttons in the user interface to prevent user interaction while performing operations related to removing ROIs.

        This method disables the model buttons and style buttons, along with other functional buttons used for handling masks, saving settings, and managing ROI deletions.

        Parameters:
        - self: The instance of the class that contains the method.

        Returns:
        - None: This method does not return any value.
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
        Toggles the mask operations by updating the layer state and adjusting saving and removal settings.

        Parameters:
        - None

        Returns:
        - None
        """
        self.update_layer()
        self.toggle_saving()
        self.toggle_removals()

    def toggle_saving(self):
        """
        Toggles the enabled state of various saving options based on the number of cells.

        This method checks the current number of cells. If the number of cells is greater than zero, it enables several saving options. If there are no cells, it disables those options.

        Parameters:
        - self: The instance of the class to which this method belongs.

        Returns:
        - None
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
        Toggles the availability of removal-related UI elements based on the number of cells.

        This method enables or disables various buttons in the user interface, such as
        the ClearButton, remcell, undo, MakeDeletionRegionButton, DeleteMultipleROIButton,
        DoneDeleteMultipleROIButton, and CancelDeleteMultipleROIButton, depending on
        the current count of cells.

        Parameters:
        - ncells: The number of cells currently present.

        Returns:
        - None: This method does not return any value.
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
        No valid docstring found.
        """
        if self.selected > 0:
            self.remove_cell(self.selected)

    def undo_action(self):
        """
        No valid docstring found.
        """
        if len(self.strokes) > 0 and self.strokes[-1][0][0] == self.currentZ:
            self.remove_stroke()
        else:
            # remove previous cell
            if self.ncells > 0:
                self.remove_cell(self.ncells)

    def undo_remove_action(self):
        """
        No valid docstring found.
        """
        self.undo_remove_cell()

    def get_files(self):
        """
        Gets the image files from a specified folder and identifies the current image file's index.

        This method retrieves all image files that match a specified mask filter from the directory containing
        the current image file described by the instance's filename. It then returns the list of image file paths
        and the index of the current image file within that list.

        Returns:
            A tuple containing:
                - A list of image file paths.
                - The index of the current file in the list of image files.
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
        Retrieves the previous image in the sequence of images.

        This method obtains the list of images and calculates the index of the previous image
        by decrementing the current index and wrapping around if necessary. It then loads the
        previous image using the appropriate image loading method.

        Returns:
            None: This method does not return a value; it triggers the loading of the previous image.
        """
        images, idx = self.get_files()
        idx = (idx - 1) % len(images)
        io._load_image(self, filename=images[idx])

    def get_next_image(self, load_seg=True):
        """
        Retrieves the next image in the sequence for processing.

        This method cycles through a list of images and loads the next image based on the current index. When the end of the list is reached, it wraps around to the beginning.

        Parameters:
            load_seg: A boolean indicating whether to load the segmentation associated with the image.

        Returns:
            None
        """
        images, idx = self.get_files()
        idx = (idx + 1) % len(images)
        io._load_image(self, filename=images[idx], load_seg=load_seg)

    def dragEnterEvent(self, event):
        """
        No valid docstring found.
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handles the drop event of files, loading data based on the file extension.

        This method processes the files dropped onto the application, checks the file extension,
        and calls the appropriate loading function for either segmentation or image data.

        Parameters:
        - event: The drop event containing mime data with URLs of the dropped files.

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
        Toggles the visibility of masks and outlines based on the state of checkbox selections.

        This method updates boolean flags corresponding to different visual elements in the
        application based on whether their associated checkboxes are checked. It also manages
        the state of a graphical layer by adding or removing it from the viewport.

        Parameters:
        - None

        Returns:
        - None: This method does not return any value.
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
        Creates a viewbox for displaying images and drawing on them.

        This method initializes a ViewBox with specific settings, adds image and drawing layers to it,
        and configures the interactive elements of the viewbox.

        Parameters:
        - parent: The parent widget or component that hosts the viewbox.
        - win: The window or layout to which the viewbox will be added.

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
        No valid docstring found.
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
        Sets the brush size based on the current index of the BrushChoose widget and updates the drawing layer.

        This method calculates the brush size by taking the current index of the BrushChoose and multiplying it by 2, then adding 1. If a layer is loaded, it updates the draw kernel of the layer with the new brush size and refreshes the layer.

        Parameters:
        None

        Returns:
        None
        """
        self.brush_size = self.BrushChoose.currentIndex() * 2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_layer()

    def clear_all(self):
        """
        Clears all selections and resets the state of the object.

        This method resets various attributes related to selection and pixel data,
        preparing the object for a fresh state. It handles different scenarios based
        on whether a restoration process is initiated.

        Parameters:
        - restore: A flag indicating if the restoration process is activated.

        Returns:
        - None: This method does not return any value.
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
        Selects a cell based on its index and updates the relevant properties and visual representations.

        This method updates the currently selected cell index, generates a mask for the selected cell,
        removes small labels from the mask, calculates measurements for the selected cell, and updates
        the visual layer with the corresponding attributes.

        Parameters:
        - idx: The index of the cell to be selected.

        Returns:
        - None: The method does not return a value.
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
        Selects a cell from a multi-select layer and updates its visual representation.

        This method updates the color of the selected cell in the current layer, reflecting the selection by changing its opacity.

        Parameters:
            idx: The index of the cell to be selected.

        Returns:
            None
        """
        if idx > 0:
            z = self.currentZ
            self.layerz[self.cellpix[z] == idx] = np.array(
                [255, 255, 255, self.opacity]
            )
            self.update_layer()

    def unselect_cell(self):
        """
        Unselects the currently selected cell, resetting its properties and updating the display layer.

        This method checks if a cell is currently selected, and if so, it reverts the cell's appearance by updating its color and opacity based on its index. Additionally, if outlines are enabled, it updates the outline color for that cell. Finally, it resets the selected cell index to zero.

        Parameters:
        - None

        Returns:
        - None: This method does not return a value.
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
        Unselects a cell in the current layer by updating its color and opacity.

        This method modifies the visual representation of a cell identified by the given index. If outlines are enabled, it also updates the outline color of the cell.

        Args:
            idx: The index of the cell to unselect.

        Returns:
            None: This method does not return any value.
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
        Removes one or more cells from the current state and updates the necessary properties.

        This method handles the removal of cells by first determining if a single index or multiple indices have been provided. It ensures that cells are removed in reverse order to maintain proper indexing. After removing the cells, it updates the number of cells and adjusts UI elements accordingly.

        Args:
            idx: A single index or a list of indices representing the cells to be removed.

        Returns:
            None: This method does not return any value.
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
        No valid docstring found.
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
        Removes region cells and creates a new ROI (Region of Interest) in the center of the view.

        This method clears the list of cells marked for removal, disables related buttons,
        and then generates a new ROI that is half the size of the current view.
        The ROI is positioned at the center of the view and is added to the displayed items.

        Parameters:
        - None

        Returns:
        - None
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
        Deletes multiple cells by enabling relevant buttons and preparing the interface for deletion.

        This method performs several actions to prepare for the deletion of multiple cells. It unselects any currently selected cell, disables buttons related to removing regions of interest (ROIs), and enables buttons necessary for executing or canceling the deletion process. The method also sets a flag indicating multiple deletions are in progress.

        Parameters:
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
        Stops the removal process of multiple cells, resets relevant flags, and updates the GUI buttons accordingly. It processes the list of cells marked for removal and triggers their deletion, as well as handles any associated region of interest (ROI) objects.

        Parameters:
        - None

        Returns:
        - None
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
        Merges two selected cells in a multi-layer cell representation.

        This method updates the cell representation by merging the selected cell with another previously selected cell. It handles the pixel data and redraws the affected regions accordingly. Additionally, it saves the changes and disables the undo/redo options after the operation.

        Parameters:
        - idx: The index of the cell to merge with the previously selected cell.

        Returns:
        None
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
        Restores a previously removed cell to the current state.

        This method checks if there is a removed cell and, if so, redraws it to the active layer,
        increments the count of cells, updates the corresponding lists and states, and saves
        the changes. The removed cell data is cleared after restoration.

        Parameters:
        - None

        Returns:
        None
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
        Removes a specified stroke from the current drawing layer and updates the layer accordingly.

        Args:
            e_points: Indicates whether to remove endpoint connections (default is True).
            stroke_ind: The index of the stroke to be removed (default is -1).

        Returns:
            None
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
        Handles mouse click events for plotting.

        This method responds to mouse click events, specifically handling left button clicks that are not combined with shift or alt modifiers. If a double-click occurs, it attempts to set the y-range for the plotting area to defined limits.

        Parameters:
            event: The mouse event that contains information about the type of click and modifiers.

        Returns:
            None: This method does not return a value.
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

        This method clears the currently selected cells and completes the removal process
        for multiple cells that were previously selected.

        Parameters:
        None

        Returns:
        None
        """
        self.clear_multi_selected_cells()
        self.done_remove_multiple_cells()

    def clear_multi_selected_cells(self):
        """
        Clears the selection of multiple cells by unselecting each cell in the removing cells list.

        This method iterates over the list of cells that are marked for removal and calls
        the unselect function on each. After unselecting the cells, it clears the list of
        removing cells.

        Returns:
            None: This method does not return a value.
        """
        # unselect all previously selected cells:
        for idx in self.removing_cells_list:
            self.unselect_cell_multi(idx)
        self.removing_cells_list.clear()

    def add_roi(self, roi):
        """
        Adds a Region of Interest (ROI) item to the p0 object and stores a reference to it for future removal.

        Args:
            roi: The Region of Interest object to be added.

        Returns:
            None
        """
        self.p0.addItem(roi)
        self.remove_roi_obj = roi

    def remove_roi(self, roi):
        """
        Removes a region of interest (ROI) from the graphical interface and clears related selections.

        This method clears any multi-selected cells, asserts that the given ROI is the currently tracked
        ROI for removal, and then removes it from the view. The removal process also resets the state
        indicating that a region is being removed.

        Parameters:
        roi: The region of interest object to be removed from the graphical interface.

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
        Updates the selection of cells based on the region of interest (ROI) changed by the user.

        This method determines the bounds of the ROI, finds all unique cell indices that overlap with this region,
        and updates the current selection by clearing previous selections and selecting new ones based on the overlap.

        Args:
            roi: The region of interest which contains the position and size to determine the overlapping cells.

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
        Handles events when the mouse is moved over a specific position, retrieving the items present at that location within the scene.

        Parameters:
            pos: The position of the mouse event.

        Returns:
            None
        """
        items = self.win.scene().items(pos)

    def color_choose(self):
        """
        Selects a color from a dropdown menu and updates the plot accordingly.

        This method retrieves the current index of the RGB dropdown menu and sets
        the view index to zero. It then updates the associated plot to reflect
        the color selection made by the user.

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
        Updates the plot based on the current view and selected parameters. This method handles different cases for displaying images, including scaling and color mapping, and updates the corresponding UI elements.

        Parameters:
        - view: The index of the current view selected from the dropdown.
        - currentZ: The current slice index of the 3D stack being visualized.
        - restore: A flag indicating whether to restore previous settings.
        - flows: A list of flow data for manipulation between views.
        - stack: A 3D array representing the original image stack.
        - stack_filtered: A 3D array representing the filtered image stack.
        - nchan: The number of channels in the image data.
        - color: The index of the color channel currently displayed.
        - saturation: A 2D array defining the saturation levels for each channel.

        Returns:
        - None
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
        Updates the layer by setting the image and refreshing the display.

        This method checks the state of certain flags and sets the image for the layer accordingly. It also updates the region of interest (ROI) count and refreshes the user interface to reflect changes.

        Args:
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
        Updates the displayed count of Regions of Interest (ROIs) in the user interface.

        This method sets the text of the ROI count display to show the current
        number of ROIs based on the value of the `ncells` attribute.

        Args:
            None

        Returns:
            None
        """
        self.roi_count.setText(f"{self.ncells} ROIs")

    def add_set(self):
        """
        Adds a set of points to the current cell, creating a mask based on the provided point set.

        This method checks if the current point set is valid and has an appropriate size. If it passes the checks, it processes the points, applies a color mask, and updates internal states accordingly. If the point set is too small, it raises an error message.

        Parameters:
          - self: The instance of the class.

        Return:
          None: This method does not return a value but modifies the internal state of the object.
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
        Adds a mask to the drawing area based on provided stroke points.

        This method processes a series of stroke points to create a mask
        representing the drawn area. It handles potential overlaps with existing
        cells and adjusts the mask accordingly. It also logs the time of execution
        and tracks changes made during the operation.

        Parameters:
            points: A list of arrays, where each array contains stroke points
                    represented as coordinate tuples.
            color: A tuple representing the RGB color value used for the mask (default is (100, 200, 50)).
            dense: A boolean indicating whether to produce a dense outline (default is True).

        Returns:
            A list containing the median coordinates of the drawn mask.
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
        Computes the scale for a graphical representation based on the given diameter.

        This method calculates the radius and creates an array to represent the scale
        for graphical visualization. It also adjusts the y and x ranges of the plot based
        on the calculated dimensions.

        Parameters:
        - Diameter: A user-provided value representing the diameter.
        - Ly: The height dimension for the visualization.
        - Lx: The width dimension for the visualization.

        Returns:
        - None
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
        Updates the scale of the visual representation based on the current radii values.

        This method recalculates the scale, sets the image and levels for the scale,
        and then displays the updated visual representation in the window.

        Parameters:
        - None

        Returns:
        - None
        """
        self.compute_scale()
        self.scale.setImage(self.radii, autoLevels=False)
        self.scale.setLevels([0.0, 255.0])
        self.win.show()
        self.show()

    def redraw_masks(self, masks=True, outlines=True, draw=True):
        """
        Redraws the masks on the layer.

        This method updates the visual representation of the masks based on the current
        state of the layer. It is responsible for ensuring that the masks are drawn
        accurately to reflect any changes in the underlying data.

        Parameters:
            outlines: Whether to draw outlines of the masks.
            draw: A flag indicating whether to perform the drawing operation.

        Returns:
            None
        """
        self.draw_layer()

    def draw_masks(self):
        """
        Draws the masks by invoking the draw_layer method.

        Parameters:
        None

        Returns:
        None
        """
        self.draw_layer()

    def draw_layer(self):
        """
        Draws a graphical layer based on the current state of the drawing application, including masks and outlines.

        This method adjusts the layer's dimensions based on the resizing setting and updates the pixel data for filling in colors and opacity. It handles the drawing of strokes and selection highlighting if applicable.

        Parameters:
        - resize: Indicates whether the layer dimensions should be adjusted.
        - masksOn: A flag that determines if masks should be drawn on the layer.
        - outlinesOn: A flag that indicates if outlines should be applied to the drawn content.
        - restore: The state of the restoration setting affecting how the pixel data is handled.
        - opacity: The transparency level applied to the drawn content.
        - currentZ: The index that represents the currently active layer to be drawn.
        - cellpix: The pixel data representing which cell colors to use.
        - cellcolors: An array containing the color data for each cell.
        - selected: Indicates the currently selected item or cell.
        - strokes: A collection of stroke data to be drawn on the layer.
        - outpix: The pixel data for outlines to be drawn on the layer.
        - outcolor: The color used for drawing outlines.

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
        Sets the style of the denoise buttons based on the current restoration settings.

        This method iterates through the denoise buttons and applies a specific style
        to each button. The style is determined by whether the button's key is in the
        restore list or if the restore is set to None.

        Parameters:
        - None

        Returns:
        - None: This method does not return any value.
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
        Sets the normalization parameters for processing images based on the provided
        configuration and default values.

        Args:
            normalize_params: A dictionary containing the normalization parameters.
                              This may include keys such as 'percentile',
                              'sharpen_radius', 'smooth_radius', 'tile_norm_blocksize',
                              'tile_norm_smooth3D', 'norm3D', and 'invert'.

        Returns:
            None: This method modifies the 'normalize_params' dictionary in place
                  and does not return a value.
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
        Checks and validates the given percentile parameters for normalization.
        If the parameters are invalid, defaults are set.

        Args:
            percentile: A list containing two elements representing lower and upper percentiles.

        Returns:
            A list of validated percentile values.
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

        This method ensures that the provided filter parameters are non-negative
        and sets the filter values in the user interface. It also checks if the
        tile size exceeds the image dimensions and disables it if necessary.

        Parameters:
        sharpen: The sharpening factor to be applied.
        smooth: The smoothing factor to be applied.
        tile_norm: The tile normalization size.
        smooth3D: The 3D smoothing factor to be applied.
        norm3D: A flag indicating whether to use 3D normalization.
        invert: A flag indicating whether to invert the image colors.

        Returns:
        A tuple containing the validated filter parameters:
        (sharpen, smooth, tile_norm, smooth3D, norm3D, invert).
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
        Generates normalization parameters based on user input.

        This method collects normalization settings for data processing, including percentile values and various filter parameters, if applicable. It checks input validity and returns a dictionary of the normalization parameters used in the application.

        Parameters:
        - percentile: A list containing two float values representing the lower and upper percentiles.
        - norm3D: A boolean indicating whether 3D normalization is enabled.
        - sharpen: A float value representing the radius for sharpening filter.
        - smooth: A float value representing the radius for smoothing filter.
        - tile_norm: A float value specifying the block size for tile normalization.
        - smooth3D: A float value for 3D smoothing.
        - invert: A boolean indicating whether to invert the filter effects.

        Returns:
        - A dictionary containing the normalization parameters that include percentile, norm3D, and other filter settings if applicable.
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
        Computes the saturation levels of an image based on normalization parameters and channels.

        This method processes the original image stack, applies normalization and filtering based on specified parameters,
        and computes saturation levels for each channel. It handles grayscale images differently and provides options
        for inverting the image. The resulting saturation levels are stored in the object's `saturation` attribute.

        Args:
            return_img: A boolean indicating whether to return the processed image.

        Returns:
            None: The method modifies the object's attributes but does not return any value.
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
        Chooses specific channels from an image based on user selection.

        This method processes a multi-dimensional image according to the current selections
        in the `ChannelChoose` interface elements. If there are multiple channels available,
        it can either compute the mean of all channels or select specific channels based on
        the user’s choice.

        Parameters:
        - image: The input image which may have multiple channels.
        - self: An instance of the class containing ChannelChoose attribute.

        Returns:
        A modified image which is either the mean of the channels or a selection
        of channels based on the user's choice from the `ChannelChoose` interface.
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
        Gets the path of the current model based on the selected option.

        If a custom model is chosen, it retrieves the model name from the user interface
        and constructs the path accordingly. If a pre-defined model is selected, it
        retrieves the model name from a list and gets the corresponding path.

        Parameters:
        custom: Indicates whether a custom model is being used. If True, the model name
                is sourced from the user interface. If False, the model name is retrieved
                from a predefined list.

        Returns:
        str: The file path to the current model.
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
        Initializes a model based on the specified model name or custom settings.

        This method checks the provided model name and either retrieves the appropriate model path or raises an error if the model name is not specified correctly. If a valid model name is provided, it initializes the Cellpose model with the corresponding parameters.

        Args:
            model_name: The name of the model to be initialized, which can be a specific model name or None.
            del_name: An optional name that can be used to delete a specific model (default is None).
            custom: A boolean indicating whether to use custom settings for the model.

        Returns:
            None: This method does not return any value.
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


        Adds a model to the I/O system.



        This method processes the addition of a model within the I/O system. It does not take any parameters.

        Once executed, the model is integrated into the underlying structure.



        Returns:

            None: This method does not return any value.

        """
        io._add_model(self)
        return

    def remove_model(self):
        """
        Removes the current model from the I/O subsystem.

        This method calls the internal function to remove the model from the
        I/O system, ensuring that resources associated with the model are
        properly released.

        Returns:
            None: This method does not return any value.
        """
        io._remove_model(self)
        return

    def new_model(self):
        """
        Trains a model if the data is not 3D.

        This method checks if the training data is 3D and proceeds to train the model using the available training data. If the training process is canceled by the user, an appropriate message will be printed.

        Returns:
            None: This method does not return any value.
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
        Trains a new model using specified training parameters and data.

        This method handles the initialization and training of a Cellpose model,
        including setting up logging, parameters for training, and saving
        the model and training losses. It prepares the model for processing
        and computes segmentation results.

        Args:
            normalize_params: Parameters for normalization. If not provided,
                default normalization parameters will be used.

        Returns:
            A tuple containing the path to the newly trained model and
            the training losses recorded during training.
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
        Compute and execute image restoration based on the specified restoration method.

        This method checks if a restoration process should be performed and manages
        the selection of denoising models or saturation computation based on the
        restoration parameters. It adjusts settings according to the specified dataset
        and restoration type, including upsampling adjustments.

        Parameters:
        self.restore: A string indicating the type of restoration to be applied.
        self.logger: A logger instance used to log information messages.
        self.DenoiseChoose: A widget that allows selection of the denoising model.
        self.Diameter: A widget for displaying the computed diameter based on upsampling ratio.
        self.ratio: A numerical value used in the upsampling calculation.

        Returns:
        None: This method does not return a value.
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
        Gets the flow threshold and cell probability threshold values.

        This method retrieves the threshold values from the UI elements,
        converts them to float, and handles exceptions by setting default
        values if the conversion fails.

        Returns:
            A tuple containing the flow threshold and cell probability threshold.
            If the flow threshold is zero or NZ is greater than 1, the flow
            threshold will be None.
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
        Computes the probability masks based on flow and cell probability thresholds.

        This method evaluates thresholds for flow and cell probability to generate masks used in further analysis. If the recompute_masks flag is enabled, it logs information about the thresholds being used, calculates the masks, and updates the GUI accordingly.

        Parameters:
        - recompute_masks: A flag indicating whether to recompute the masks.
        - flows: A list containing flow data required for mask computation.
        - cellpix: An array representing cell pixel data.
        - OCheckBox: A checkbox control for user interface interactions.
        - MCheckBox: A checkbox control for user interface interactions.
        - logger: A logging object used for providing runtime information.

        Returns:
        - None: This method does not return any value but updates the GUI with the computed masks.
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
        Computes the denoising model based on user-selected parameters and updates the progress and internal states accordingly.

        Args:
            f: The function or data to process, which triggers the denoising operation.
            model_type: The type of model to be used for denoising.

        Returns:
            None: This method does not return a value. It modifies internal states and UI components directly.
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
        Computes segmentation of the input data using a specified model and various parameters.

        This method initializes the model, processes the input data, and generates segmentation masks and flow data based on the configured parameters. It provides progress updates and handles exceptions gracefully during computation.

        Args:
            custom: A boolean indicating whether to use a custom model.
            model_name: The name of the model to be used for segmentation.
            load_model: A boolean that specifies whether to load the model before computation.

        Returns:
            None: The method modifies class attributes directly and does not return any value.
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
