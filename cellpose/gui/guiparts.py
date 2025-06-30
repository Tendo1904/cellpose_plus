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
    Generates a stylesheet string for customizing the appearance of UI components.

    This method creates and returns a string containing the stylesheet for various
    Qt widgets such as QToolTip, QComboBox, QScrollArea, QGroupBox, and QPushButton.
    The returned stylesheet defines the visual attributes like background color, text
    color, border properties, and other styling elements of these widgets.

    Returns:
        str: A string representing the stylesheet for the UI components.
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
        Initializes an instance of the ImageDraw class.

        This constructor sets up the initial state for the ImageDraw object, configuring
        various attributes related to image processing and drawing.

        Args:
            image: The image to be drawn on, if any.
            viewbox: The viewbox dimensions for the drawing context, if applicable.
            parent: The parent object that this instance is associated with.
            **kargs: Additional keyword arguments to handle other options or configurations.

        Returns:
            None
        """
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        """
        Sets up the color palette for the application interface.

        This method configures various color settings used in the application
        window, including background colors, text colors, button colors, and
        highlight colors for different states including the disabled state.

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
    Creates a user interface for selecting channels used in segmentation.

    This method sets up two combo boxes for channel selection, populating them
    with predefined options for channel colors. It also creates labels with tooltips
    to guide the user in selecting the appropriate channels for cytoplasm and nuclei
    segmentation.

    Returns:
        A tuple containing:
            - A list of QComboBox objects for channel selection.
            - A list of QLabel objects for labeling the combo boxes.
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
    ModelButton is a class that represents a button associated with a specific model, providing an interface for user interaction.

    Attributes:
    - parent: The parent widget that holds this button.
    - model_name: The name of the model associated with this button.
    - text: The text to display on the button.
    - is_pressed: A flag indicating whether the button is currently pressed.

    Methods:
    - __init__
    - press

    The __init__ method initializes an instance of the ModelButton class with the specified parent, model name, and display text.
    The press method is responsible for handling the logic when the button is pressed, triggering any associated actions or updates.
    """

    def __init__(self, parent, model_name, text):
        """
        Initializes a new instance of the class.

        This constructor sets up the initial state of the object, including its parent, border properties,
        and various configuration settings related to view behavior.

        Args:
            parent: The parent object that this instance is associated with.
            border: Specifies the border settings for the view.
            lockAspect: A boolean indicating whether to lock the aspect ratio.
            enableMouse: A boolean indicating whether mouse interaction is enabled.
            invertY: A boolean indicating whether the Y-axis should be inverted.
            enableMenu: A boolean indicating whether to enable the menu.
            name: A name for the view.
            invertX: A boolean indicating whether the X-axis should be inverted.

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
        Press method for applying filtering or denoising operations based on the model type.

        This method checks the model type and applies the necessary processing on the provided parent object. If the model type is set to "filter", it verifies the filter settings and restores them accordingly. For other model types, it computes the denoise model. If no valid model type is set, it clears the restore settings.

        Args:
            parent: An object that contains methods for restoring, normalizing parameters, computing saturation, and handling restore button states.

        Returns:
            None: This method does not return a value.
        """
        parent.compute_segmentation(model_name=self.model_name)


class DenoiseButton(QPushButton):
    """No valid docstring found."""

    def __init__(self, parent, text):
        """
        Initializes the TrainHelpWindow class, setting up the user interface and loading help text.

        Args:
            parent: The parent widget for this window.

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
        No valid docstring found.
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
    This class represents a training window in a graphical user interface for configuring
    and managing the training parameters of a machine learning model.

    Attributes:
    - parent: The parent widget that this window belongs to.
    - model_strings: A list of model names to choose from.

    Methods:
    - __init__
    - accept

    The __init__ method initializes the training settings window, setting up the interface
    for model and channel selection as well as other training configurations. The accept
    method processes and accepts the user's input for the training parameters, updating
    the parent object accordingly.
    """

    def __init__(self, parent, model_strings):
        """
        No valid docstring found.
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
        Accepts the training parameters from the user input and updates the parent object's training configuration.

        Args:
            parent: The parent object that will hold the training parameters.

        Returns:
            None
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
    ExampleGUI is a class for creating a graphical user interface (GUI) that displays an image and allows user interactions.

    Attributes:
    - window: The main window of the GUI.
    - image_label: A label that displays the loaded image.

    Methods:
    - __init__: Initializes an instance of the ExampleGUI class.

    This method sets up the main window of the GUI, including its geometry, title, and layout.
    It loads an image from a specified path and adds it to the window.
    Args:
        parent: The parent widget for this GUI window. If not provided, it defaults to None.

    Returns:
        None
    """

    def __init__(self, parent=None):
        """
        Initializes an instance of the ExampleGUI class.

        This method sets up the main window of the GUI, including its geometry, title, and layout.
        It loads an image from a specified path and adds it to the window.

        Args:
            parent: The parent widget for this GUI window. If not provided, it defaults to None.

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
    HelpWindow is a GUI class that displays help information to users in a dedicated window.

    Attributes:
    - window: The window object for displaying help content.
    - content: The content to be displayed in the help window.

    Methods:
    - __init__:
    """

    def __init__(self, parent=None):
        """
        Initializes the training settings window for the model.

        This method sets up the graphical user interface for configuring training parameters,
        including model selection, channel selection, and other training settings.

        Args:
            parent: The parent widget that this window belongs to.
            model_strings: A list of model names to choose from.

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
    TrainHelpWindow is a class that provides a user interface window for displaying help information related to the training process.

    Attributes:
    - help_text: The text that contains the help information.
    - parent_widget: The parent widget that contains this help window.

    Methods:
    - __init__:
        Initializes the TrainHelpWindow class, setting up the user interface and loading help text.

    Args:
        parent: The parent widget for this window.

    Returns:
        None
    """

    def __init__(self, parent=None):
        """
        No valid docstring found.
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
    ViewBoxNoRightDrag is a custom view box class that disables right mouse dragging to prevent panning, while allowing other mouse interactions and zooming functionalities.

    Attributes:
    - parent: The parent object that this instance is associated with.
    - border: Specifies the border settings for the view.
    - lockAspect: Indicates whether to lock the aspect ratio.
    - enableMouse: Indicates whether mouse interaction is enabled.
    - invertY: Indicates whether the Y-axis should be inverted.
    - enableMenu: Indicates whether to enable the menu.
    - name: A name for the view.
    - invertX: Indicates whether the X-axis should be inverted.

    Methods:
    - __init__: Initializes a new instance of the class, setting up its initial state and configuration.
    - keyPressEvent: Captures key presses for zooming functionality in the current view box.
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
        Initializes an instance of the class, setting up the button's initial state and properties.

        Args:
            parent: The parent widget that holds this button.
            model_name: The name of the model associated with this button.
            text: The text to display on the button.

        Returns:
            None
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
        Initializes a new instance of the class and sets up the user interface.

        This constructor calls the parent class's initializer and then invokes
        the setup method to configure the UI components.

        Parameters:
        None

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
        Handles mouse click events for drawing and selecting regions in a graphical user interface.
        It manages strokes for drawing and interactions with cells based on mouse buttons and modifiers.

        Args:
            ev: The event object that contains information about the mouse click, including its position and the button clicked.

        Returns:
            None
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
        No valid docstring found.
        """
        ev.ignore()
        return

    def hoverEvent(self, ev):
        """
        Handles the hover event for drawing operations. This method processes the event based on the current state of the drawing tool, allowing the user to continue a stroke or end it based on specific conditions.

        Args:
            ev: The event object containing the position of the cursor and other relevant information.

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
        Creates a scatter plot item at the specified position and adds it to the parent item.

        Args:
            pos: The position where the scatter plot item will be created.

        Returns:
            None
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
        No valid docstring found.
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
        Ends the current stroke by removing the associated scatter item, appending the current stroke to the strokes list if it hasn't been appended yet, and storing points that are part of the stroke's outline. If autosave is enabled, it adds the current set of points to the parent.

        Parameters:
        - None

        Returns:
        - None
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
        Handles events related to tablet input.

        Args:
            ev: The event object containing information about the tablet actions, such as device, pointer type, and pressure.

        Returns:
            None: This method does not return any value.
        """
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        """
        Draws an image at a specified position using a stroke mask.

        This method updates the image with the stroke mask at the given position,
        considering the dimensions of the drawing kernel and the image boundaries.
        It also logs the stroke information for the current drawing operation.

        Args:
            pos: The position where the image will be drawn, given as a coordinate.
            ev: An optional event that may contain additional information for drawing.

        Returns:
            None: This method does not return a value.
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

        Sets the drawing kernel for the image processing.

        This method creates a square kernel of the specified size, which is used for drawing operations.
        It initializes the drawKernel and its center, as well as the onmask, offmask, and opamask
        for further processing.

        Args:
            kernel_size: The size of the kernel to be created.

        Returns:
            None
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
