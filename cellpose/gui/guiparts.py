"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtGui import QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QRadioButton,
    QWidget,
    QDialog,
    QButtonGroup,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QGridLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QComboBox,
    QCheckBox,
)
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph import Point
import numpy as np
import pathlib, os


def stylesheet():
    """
    Generates the CSS stylesheet for the application.

        This method returns a string that contains the CSS styles defined for various
        UI components in the application. The styles include configurations for
        tooltips, combo boxes, scroll areas, group boxes, and buttons to ensure a
        consistent appearance throughout the UI.

        Returns:
            str: A string representing the CSS stylesheet for the application UI components.
    """
    return """
        QToolTip { 
                            background-color: black; 
                            color: white; 
                            border: black solid 1px
                            }
        QComboBox {color: white;
                    background-color: rgb(40,40,40);}
                    QComboBox::item:enabled { color: white;
                    background-color: rgb(40,40,40);
                    selection-color: white;
                    selection-background-color: rgb(50,100,50);}
                    QComboBox::item:!enabled {
                            background-color: rgb(40,40,40);
                            color: rgb(100,100,100);
                        }
        QScrollArea > QWidget > QWidget
                {
                    background: transparent;
                    border: none;
                    margin: 0px 0px 0px 0px;
                } 
                           
        QGroupBox 
            { border: 1px solid white; color: rgb(255,255,255);
                           border-radius: 6px;
                            margin-top: 8px;
                            padding: 0px 0px;}            
                           
        QPushButton:pressed {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:!pressed {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:disabled {Text-align: center; 
                             background-color: rgb(30,30,30);
                             border-color: white;
                              color:rgb(80,80,80);}
                               QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
                        
        """


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
    (from pykilosort/kilosort4)
    """

    def __init__(self):
        """
        Initializes the class and sets up the GUI components.

            This method calls the parent class's initializer and then
            executes the setup method to configure the user interface.

            Returns:
                None: This method does not return a value.
        """
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        """
        Sets up the color scheme for the application's GUI.

            This method configures various color roles in the application's GUI,
            setting colors for different UI elements to create a cohesive and visually
            appealing interface.

            Parameters:
                None

            Returns:
                None
        """
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(34, 27, 24))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(
            QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(128, 128, 128)
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )


def create_channel_choose():
    """
    Creates a selection interface for channel choices in segmentation.

        This method generates two combo boxes for selecting channels related to
        segmentation. The first combo box allows the user to choose a primary
        channel for segmentation (e.g., cytoplasm or nuclei), and the second combo
        box is optional for selecting a secondary channel. Tooltips are provided for
        better understanding of the selections.

        Returns:
            A tuple containing two elements:
                - A list of QComboBox instances for channel selections.
                - A list of QLabel instances corresponding to channel labels.
    """
    # choose channel
    ChannelChoose = [QComboBox(), QComboBox()]
    ChannelLabels = []
    ChannelChoose[0].addItems(["gray", "red", "green", "blue"])
    ChannelChoose[1].addItems(["none", "red", "green", "blue"])
    cstr = ["chan to segment:", "chan2 (optional): "]
    for i in range(2):
        ChannelLabels.append(QLabel(cstr[i]))
        if i == 0:
            ChannelLabels[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment"
            )
            ChannelChoose[i].setToolTip(
                "this is the channel in which the cytoplasm or nuclei exist \
            that you want to segment"
            )
        else:
            ChannelLabels[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option"
            )
            ChannelChoose[i].setToolTip(
                "if <em>cytoplasm</em> model is chosen, and you also have a \
            nuclear channel, then choose the nuclear channel for this option"
            )

    return ChannelChoose, ChannelLabels


class ModelButton(QPushButton):
    """
    A button that triggers segmentation computation based on a specified model.

    This class creates a button that, when pressed, executes a segmentation process
    using a model name. It is designed to be integrated into a user interface where
    it interacts with a parent object capable of performing the segmentation.

    Methods:
        __init__
        press

    Attributes:
        None

    __init__: Initializes the button with the given text, font, and connections,
              and prepares it for interaction.

    press: Triggers the segmentation computation using the associated model name
           on the specified parent object.
    """

    def __init__(self, parent, model_name, text):
        """
        Initializes an instance of the class.

            This method sets up the initial state of the object by configuring
            its text, font, and connections. It disables the button and connects
            it to a press event, handling specific model name adjustments.

            Args:
                parent: The parent widget that provides context and styling.
                model_name: The name of the model, which may be modified for specific cases.
                text: The text to be displayed on the button.

            Returns:
                None
        """
        super().__init__()
        self.setEnabled(False)
        self.setText(text)
        self.setFont(parent.boldfont)
        self.clicked.connect(lambda: self.press(parent))
        self.model_name = model_name if "cyto3" not in model_name else "cyto3"

    def press(self, parent):
        """
        Executes the segmentation computation on the provided parent object.

            This method calls the compute_segmentation function on the parent object,
            utilizing the model name associated with this instance.

            Args:
                parent: An object that has a compute_segmentation method, which takes
                        a model name as an argument.

            Returns:
                None: This method does not return any value.
        """
        parent.compute_segmentation(model_name=self.model_name)


class DenoiseButton(QPushButton):
    """
    A button for applying denoising or filtering operations based on a specified model type.

    This class provides functionality to initialize a button with a specific
    display text and to process filtering or denoising operations when the button
    is pressed. It interacts with a parent object to modify its state according
    to the selected model type.

    Methods:
        __init__: Initializes a new instance of the class.
        press: Processes the filtering or denoising based on the model type.

    Attributes:
        parent: The parent object that provides context and resources.
        text: The text to display and use as the model type.

    The `__init__` method sets up the initial state of the DenoiseButton, including
    its font, text, and click behavior. The `press` method handles the application
    of filtering or denoising, validating parameters and updating the parent object
    based on the specified model type.
    """

    def __init__(self, parent, text):
        """
        Initializes a new instance of the class.

            This method sets up the initial state of the object, including its
            font, text, and click behavior.

            Args:
                parent: The parent object that provides context and resources.
                text: The text to display and use as the model type.

            Returns:
                None
        """
        super().__init__()
        self.setEnabled(False)
        self.model_type = text
        self.setText(text)
        self.setFont(parent.medfont)
        self.clicked.connect(lambda: self.press(parent))

    def press(self, parent):
        """
        Processes the filtering or denoising based on the model type.

            This method checks the model type and applies filtering or denoising
            operations to the parent object. If the model type is "filter", it validates
            the normalization parameters and executes the filtering process. If the
            model type is not "none", it applies a denoising model. If the model type
            is "none", it clears any restore settings.

            Args:
                parent: The parent object that holds the state and methods for
                        computing filtering or denoising.

            Returns:
                None: This method modifies the parent object in place and does not
                      return any values.
        """
        if self.model_type == "filter":
            parent.restore = "filter"
            normalize_params = parent.get_normalize_params()
            if (
                normalize_params["sharpen_radius"] == 0
                and normalize_params["smooth_radius"] == 0
                and normalize_params["tile_norm_blocksize"] == 0
            ):
                print(
                    "GUI_ERROR: no filtering settings on (use custom filter settings)"
                )
                parent.restore = None
                return
            parent.restore = self.model_type
            parent.compute_saturation()
        elif self.model_type != "none":
            parent.compute_denoise_model(model_type=self.model_type)
        else:
            parent.clear_restore()
        parent.set_restore_button()


class TrainWindow(QDialog):
    """
    A dialog for setting up and configuring training parameters for models.

    This class provides a user interface that allows users to select models,
    channels, and training parameters for machine learning training sessions.

    Methods:
        __init__
        accept

    Attributes:
        (Attributes would be listed here if defined.)

    The __init__ method initializes the training settings dialog, allowing users
    to choose model names and configure training options. The accept method
    retrieves the current settings from the dialog, updates the parent object
    with the selected parameters, and signals that the configuration process is complete.
    """

    def __init__(self, parent, model_strings):
        """
        Initialize the training settings dialog.

            This method sets up the user interface for the training settings,
            allowing users to select models, channels, and training parameters.

            Args:
                parent: The parent widget that this dialog belongs to.
                model_strings: A list of model names for selection in the combo box.

            Returns:
                None
        """
        super().__init__(parent)
        self.setGeometry(100, 100, 900, 550)
        self.setWindowTitle("train settings")
        self.win = QWidget(self)
        self.l0 = QGridLayout()
        self.win.setLayout(self.l0)

        yoff = 0
        qlabel = QLabel("train model w/ images + _seg.npy in current folder >>")
        qlabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))

        qlabel.setAlignment(QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 2)

        # choose initial model
        yoff += 1
        self.ModelChoose = QComboBox()
        self.ModelChoose.addItems(model_strings)
        self.ModelChoose.addItems(["scratch"])
        self.ModelChoose.setFixedWidth(150)
        self.ModelChoose.setCurrentIndex(parent.training_params["model_index"])
        self.l0.addWidget(self.ModelChoose, yoff, 1, 1, 1)
        qlabel = QLabel("initial model: ")
        qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(qlabel, yoff, 0, 1, 1)

        # choose channels
        self.ChannelChoose, self.ChannelLabels = create_channel_choose()
        for i in range(2):
            yoff += 1
            self.ChannelChoose[i].setFixedWidth(150)
            self.ChannelChoose[i].setCurrentIndex(
                parent.ChannelChoose[i].currentIndex()
            )
            self.l0.addWidget(self.ChannelLabels[i], yoff, 0, 1, 1)
            self.l0.addWidget(self.ChannelChoose[i], yoff, 1, 1, 1)

        # choose parameters
        labels = ["learning_rate", "weight_decay", "n_epochs", "model_name"]
        self.edits = []
        yoff += 1
        for i, label in enumerate(labels):
            qlabel = QLabel(label)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + yoff, 0, 1, 1)
            self.edits.append(QLineEdit())
            self.edits[-1].setText(str(parent.training_params[label]))
            self.edits[-1].setFixedWidth(200)
            self.l0.addWidget(self.edits[-1], i + yoff, 1, 1, 1)

        yoff += 1
        use_SGD = "SGD"
        self.useSGD = QCheckBox(f"{use_SGD}")
        self.useSGD.setToolTip(
            "use SGD, if unchecked uses AdamW (recommended learning_rate then 0.001)"
        )
        self.useSGD.setChecked(True)
        self.l0.addWidget(self.useSGD, i + yoff, 1, 1, 1)

        yoff += len(labels)

        yoff += 1
        self.use_norm = QCheckBox(f"use restored/filtered image")
        self.use_norm.setChecked(True)
        # self.l0.addWidget(self.use_norm, yoff, 0, 2, 4)

        yoff += 2
        qlabel = QLabel(
            "(to remove files, click cancel then remove \nfrom folder and reopen train window)"
        )
        self.l0.addWidget(qlabel, yoff, 0, 2, 4)

        # click button
        yoff += 3
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(lambda: self.accept(parent))
        self.buttonBox.rejected.connect(self.reject)
        self.l0.addWidget(self.buttonBox, yoff, 0, 1, 4)

        # list files in folder
        qlabel = QLabel("filenames")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 4, 1, 1)
        qlabel = QLabel("# of masks")
        qlabel.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(qlabel, 0, 5, 1, 1)

        for i in range(10):
            if i > len(parent.train_files) - 1:
                break
            elif i == 9 and len(parent.train_files) > 10:
                label = "..."
                nmasks = "..."
            else:
                label = os.path.split(parent.train_files[i])[-1]
                nmasks = str(parent.train_labels[i].max())
            qlabel = QLabel(label)
            self.l0.addWidget(qlabel, i + 1, 4, 1, 1)
            qlabel = QLabel(nmasks)
            qlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.l0.addWidget(qlabel, i + 1, 5, 1, 1)

    def accept(self, parent):
        """
        Accepts the current configuration and sets training parameters.

            This method retrieves values from various input fields and
            sets the training parameters in the provided parent object.
            It also informs the system that the configuration is complete.

            Args:
                parent: The object that will receive the training parameters.

            Returns:
                None. The method updates the parent object directly and signals completion.
        """
        # set training params
        parent.training_params = {
            "model_index": self.ModelChoose.currentIndex(),
            "learning_rate": float(self.edits[0].text()),
            "weight_decay": float(self.edits[1].text()),
            "n_epochs": int(self.edits[2].text()),
            "model_name": self.edits[3].text(),
            "SGD": True if self.useSGD.isChecked() else False,
            "channels": [
                self.ChannelChoose[0].currentIndex(),
                self.ChannelChoose[1].currentIndex(),
            ],
            # "use_norm": True if self.use_norm.isChecked() else False,
        }
        self.done(1)


class ExampleGUI(QDialog):
    """
    A class for creating a simple graphical user interface using PyQt.

    This class sets up a user interface that displays an image within a QLabel.
    It is designed to be used as a standalone GUI application or as part of a larger
    application where GUI functionality is required.

    Methods:
        __init__: Initializes the ExampleGUI instance.

    Attributes:
        parent: The parent widget for this GUI.
        window_title: The title of the GUI window.
        layout: The layout manager for arranging widgets.
        image_label: The QLabel widget used to display an image.

    The __init__ method sets up the main components of the GUI, including the
    application window's size and title. The attributes hold references to important
    GUI elements that facilitate interaction and layout management.
    """

    def __init__(self, parent=None):
        """
        Initialize the ExampleGUI instance.

            This method sets up the graphical user interface for the ExampleGUI class,
            including the window geometry, title, and layout with a QLabel displaying
            an image.

            Args:
                parent: The parent widget for this GUI. If not provided, defaults to None.

            Returns:
                None
        """
        super(ExampleGUI, self).__init__(parent)
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowTitle("GUI layout")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        guip_path = pathlib.Path.home().joinpath(".cellpose", "cellpose_gui.png")
        guip_path = str(guip_path.resolve())
        pixmap = QPixmap(guip_path)
        label = QLabel(self)
        label.setPixmap(pixmap)
        pixmap.scaled
        layout.addWidget(label, 0, 0, 1, 1)


class HelpWindow(QDialog):
    """
    A window that displays help content to users in a readable format.

    Methods:
        __init__: Initializes the HelpWindow by setting up its geometry, title,
        and layout, as well as loading and displaying help content from an HTML file.

    Attributes:
        parent: The parent widget of the HelpWindow.
        title: The title of the HelpWindow.
        content: The content loaded from the HTML file displayed within the window.

    This class enables users to view help documentation contained in a specified
    HTML file, improving user accessibility to important information regarding
    the application.
    """

    def __init__(self, parent=None):
        """
        Initialize the HelpWindow.

            This method sets up the HelpWindow by configuring its geometry,
            title, and layout. It loads the help content from an HTML file and
            displays it within a QLabel.

            Args:
                parent: The parent widget of the HelpWindow. Defaults to None.

            Returns:
                None
        """
        super(HelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 1000)
        self.setWindowTitle("cellpose help")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath("guihelpwindowtext.html")
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class TrainHelpWindow(QDialog):
    """
    A window displaying training instructions and help content.

    This class provides a graphical user interface window that shows
    users training instructions and help information, loaded from
    an HTML file. It is designed to support users seeking guidance
    while using a training application.

    Methods:
        __init__: Initializes the TrainHelpWindow and sets up its layout.

    Attributes:
        parent: The parent widget for the window, allowing it to be embedded
                within another interface if needed.
        size: The dimensions of the window, defining how much screen
              real estate the interface occupies.
        title: The title of the window as displayed on the title bar.
        content_label: A label that displays the loaded HTML content
                       within the window.

    The __init__ method prepares the window by setting its size, title,
    and layout details, and loads the relevant content from an HTML
    file to enhance user experience and provide accessible training
    information.
    """

    def __init__(self, parent=None):
        """
        Initialize the TrainHelpWindow.

            This method sets up the main window for the training instructions,
            including its size, title, and layout. It loads the content
            from an HTML file and displays it within a label in the window.

            Args:
                parent: The parent widget of this window. If no parent is
                        specified, the window will be a top-level window.

            Returns:
                None
        """
        super(TrainHelpWindow, self).__init__(parent)
        self.setGeometry(100, 50, 700, 300)
        self.setWindowTitle("training instructions")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)

        text_file = pathlib.Path(__file__).parent.joinpath(
            "guitrainhelpwindowtext.html"
        )
        with open(str(text_file.resolve()), "r") as f:
            text = f.read()

        label = QLabel(text)
        label.setFont(QtGui.QFont("Arial", 8))
        label.setWordWrap(True)
        layout.addWidget(label, 0, 0, 1, 1)
        self.show()


class ViewBoxNoRightDrag(pg.ViewBox):
    """
    A custom view box that restricts right mouse drag interactions.

    This class provides a graphical interface component that allows users
    to visualize data with customizable settings and interaction features.

    Methods:
        __init__
        keyPressEvent

    Attributes:
        parent
        border
        lockAspect
        enableMouse
        invertY
        enableMenu
        name
        invertX

    The __init__ method initializes a ViewBox instance with various
    configuration options, such as aspect ratio locking, mouse interaction,
    and axis inversion. The keyPressEvent method captures key presses
    for zooming functionality within the view box.
    """

    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=False,
        enableMouse=True,
        invertY=False,
        enableMenu=True,
        name=None,
        invertX=False,
    ):
        """
        Initializes a ViewBox instance with optional parameters.

            This method sets up a ViewBox with various configuration options,
            allowing for customization of behavior such as aspect ratio locking,
            mouse interaction, and axis inversion.

            Args:
                parent: The parent object of the ViewBox.
                border: The border style or settings for the ViewBox.
                lockAspect: A boolean indicating whether to lock the aspect ratio.
                enableMouse: A boolean indicating whether mouse interactions are enabled.
                invertY: A boolean indicating whether to invert the Y-axis.
                enableMenu: A boolean indicating whether to enable the menu.
                name: An optional name for the ViewBox.
                invertX: A boolean indicating whether to invert the X-axis.

            Returns:
                None: This method does not return a value.
        """
        pg.ViewBox.__init__(
            self,
            None,
            border,
            lockAspect,
            enableMouse,
            invertY,
            enableMenu,
            name,
            invertX,
        )
        self.parent = parent
        self.axHistoryPointer = -1

    def keyPressEvent(self, ev):
        """
        This routine should capture key presses in the current view box.
        The following events are implemented:
        +/= : moves forward in the zooming stack (if it exists)
        - : moves backward in the zooming stack (if it exists)

        """
        ev.accept()
        if ev.text() == "-":
            self.scaleBy([1.1, 1.1])
        elif ev.text() in ["+", "="]:
            self.scaleBy([0.9, 0.9])
        else:
            ev.ignore()


class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        """
        Initializes an instance of the ImageDraw class.

            This method sets up the necessary parameters for the ImageDraw instance,
            including image handling, viewbox configuration, and linking to a parent object.

            Args:
                image: An optional image to be drawn on.
                viewbox: An optional viewbox that defines the drawing area.
                parent: An optional parent object that may contain settings relevant to the drawing.
                **kargs: Additional keyword arguments for further customization.

            Returns:
                None
        """
        super(ImageDraw, self).__init__()
        # self.image=None
        # self.viewbox=viewbox
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        # kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        """
        Handles mouse click events for the application.

            This method processes mouse click events, determining the action to take
            based on the current state of the application and the properties of the
            click event. It supports different functionalities based on the type of
            mouse button clicked, whether any modifier keys are pressed, and the
            application state regarding masks and strokes.

            Args:
                ev: The mouse event that triggered this method, containing details
                    such as the position of the click and which mouse button was
                    pressed.

            Returns:
                None: This method does not return a value, but it may modify the
                internal state of the application based on the click event.
        """
        if (
            self.parent.masksOn or self.parent.outlinesOn
        ) and not self.parent.removing_region:
            is_right_click = ev.button() == QtCore.Qt.RightButton
            if (
                self.parent.loaded
                and (
                    is_right_click
                    or ev.modifiers() & QtCore.Qt.ShiftModifier
                    and not ev.double()
                )
                and not self.parent.deleting_multiple
            ):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.parent.in_stroke = True
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            elif not self.parent.in_stroke:
                y, x = int(ev.pos().y()), int(ev.pos().x())
                if y >= 0 and y < self.parent.Ly and x >= 0 and x < self.parent.Lx:
                    if ev.button() == QtCore.Qt.LeftButton and not ev.double():
                        idx = self.parent.cellpix[self.parent.currentZ][y, x]
                        if idx > 0:
                            if ev.modifiers() & QtCore.Qt.ControlModifier:
                                # delete mask selected
                                self.parent.remove_cell(idx)
                            elif ev.modifiers() & QtCore.Qt.AltModifier:
                                self.parent.merge_cells(idx)
                            elif (
                                self.parent.masksOn
                                and not self.parent.deleting_multiple
                            ):
                                self.parent.unselect_cell()
                                self.parent.select_cell(idx)
                            elif self.parent.deleting_multiple:
                                if idx in self.parent.removing_cells_list:
                                    self.parent.unselect_cell_multi(idx)
                                    self.parent.removing_cells_list.remove(idx)
                                else:
                                    self.parent.select_cell_multi(idx)
                                    self.parent.removing_cells_list.append(idx)

                        elif self.parent.masksOn and not self.parent.deleting_multiple:
                            self.parent.unselect_cell()

    def mouseDragEvent(self, ev):
        """
        Handle mouse drag events.

            This method is called when a mouse drag event occurs. It ignores the event, preventing any default
            action from taking place when the mouse is dragged.

            Args:
                ev: The event object representing the mouse drag event.

            Returns:
                None: This method does not return any value.
        """
        ev.ignore()
        return

    def hoverEvent(self, ev):
        """
        Handles hover events for the associated widget.

            This method is triggered when a hover event occurs over the widget. It performs actions based on
            whether a stroke is currently in progress. If a stroke is active, the method continues to draw
            at the position of the event. If the event occurs at the start of the stroke, it calls
            the end_stroke method. If no stroke is in progress, it accepts right-click events.

            Args:
                ev: The event associated with the hover, providing information such as the cursor position.

            Returns:
                None: This method does not return a value.
        """
        # QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    # self.parent.in_stroke = False
                    self.end_stroke()
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            # ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        """
        Create and add a scatter plot item at the specified position.

            This method initializes a scatter plot item with a specific pen color
            and size, and adds it to the parent item's plot at the given position.

            Args:
                pos: The position object containing x and y coordinates where
                     the scatter plot item will be created.

            Returns:
                None: This method does not return a value.
        """
        self.scatter = pg.ScatterPlotItem(
            [pos.x()],
            [pos.y()],
            pxMode=False,
            pen=pg.mkPen(color=(255, 0, 0), width=self.parent.brush_size),
            size=max(3 * 2, self.parent.brush_size * 1.8 * 2),
            brush=None,
        )
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        """
        Determines if the current stroke has returned to the starting point.

            This method evaluates the current stroke's distance from the starting point
            to ascertain if it has returned to the start after leaving it. It utilizes
            thresholds based on the brush size to define the conditions for 'left' and
            'returned'.

            Args:
                pos: The current position which may be used in determining stroke state.

            Returns:
                A boolean indicating whether the stroke has returned to the start point
                after being determined to have left the starting region.
        """
        thresh_out = max(6, self.parent.brush_size * 3)
        thresh_in = max(3, self.parent.brush_size * 1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (
                ((stroke[1:, 1:] - stroke[:1, 1:][np.newaxis, :, :]) ** 2).sum(axis=-1)
            ) ** 0.5
            dist = dist.flatten()
            # print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4, first_left + 1) :] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        """
        Finalize the current stroke and update stroke-related data.

            This method removes the current scatter item from the parent,
            appends the current stroke to the list of strokes if it hasn't
            been appended yet, and processes the stroke data for later use.
            It also handles the autosave functionality based on the
            state of the parent.

            Parameters:
                None

            Returns:
                None
        """
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.strokes.append(self.parent.current_stroke)
            self.parent.stroke_appended = True
            self.parent.current_stroke = np.array(self.parent.current_stroke)
            ioutline = self.parent.current_stroke[:, 3] == 1
            self.parent.current_point_set.append(
                list(self.parent.current_stroke[ioutline])
            )
            self.parent.current_stroke = []
            if self.parent.autosave:
                self.parent.add_set()
        if (
            len(self.parent.current_point_set)
            and len(self.parent.current_point_set[0]) > 0
            and self.parent.autosave
        ):
            self.parent.add_set()
        self.parent.in_stroke = False

    def tabletEvent(self, ev):
        """
        Handles tablet events.

            This method processes events generated by tablet devices, such as
            pen input. It is intended to be overridden or implemented to
            handle specific tablet event logic.

            Args:
                ev: The event object generated by the tablet device, which
                    includes details such as device information, pointer type,
                    and pressure sensitivity.

            Returns:
                None: This method does not return a value.
        """
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        """
        Draws a stroke on the image at a specified position.

            This method modifies the current image by applying a stroke mask
            at the given position. It calculates the area of effect using the
            stroke's kernel and adjusts the coordinates to ensure that they
            remain within the bounds of the image.

            Args:
                pos: The position where the stroke should be drawn, represented
                      as a coordinate object with 'y' and 'x' methods.
                ev: An optional event argument that can provide additional
                    context or information related to the drawing operation.

            Returns:
                None: This method updates the image in place and does not
                return a value.
        """
        mask = self.strokemask
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
        kcent = kc.copy()
        if tx[0] <= 0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx = sx
            kcent[0] = 0
        if ty[0] <= 0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= self.parent.Lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([self.parent.currentZ, x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        """
        Sets the drawing kernel and related masks.

            This method generates a square kernel of specified size filled with ones,
            which is used for drawing operations. It computes the center of the kernel
            and creates two masks: a red mask and a stroke mask. These masks are
            constructed by concatenating the kernel with designated opacity values for
            visual representation.

            Args:
                kernel_size: The size of the kernel to be created (should be an odd integer).

            Returns:
                None: This method modifies the instance attributes directly and does not return a value.
        """
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2)),
        ]
        onmask = 255 * kernel[:, :, np.newaxis]
        offmask = np.zeros((bs, bs, 1))
        opamask = 100 * kernel[:, :, np.newaxis]
        self.redmask = np.concatenate((onmask, offmask, offmask, onmask), axis=-1)
        self.strokemask = np.concatenate((onmask, offmask, onmask, opamask), axis=-1)
