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
    Generates a stylesheet for a graphical user interface.

    This method returns a string that contains CSS-like styling rules for various
    Qt widgets. The stylesheet enhances the appearance of Tooltips, ComboBoxes,
    ScrollAreas, GroupBoxes, and PushButtons in the application.

    Returns:
        str: A string containing the styling rules for the GUI elements.
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
        Initializes a new instance of the class, setting up the view box with specified parameters.

        Args:
            parent: The parent object to which this instance belongs.
            border: The border settings for the view box.
            lockAspect: A boolean flag to lock the aspect ratio.
            enableMouse: A boolean flag to enable mouse interactions.
            invertY: A boolean flag to invert the Y-axis.
            enableMenu: A boolean flag to enable the menu.
            name: An optional name for the instance.
            invertX: A boolean flag to invert the X-axis.

        Returns:
            None
        """
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        """
        Sets up the color palette for the application interface.

        This method configures various color settings for the application's GUI elements,
        defining colors for windows, text, buttons, and tooltips to ensure consistent visual
        appearance.

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
    Creates a channel selection interface with two QComboBox widgets for user input.

    This method initializes two combo boxes for selecting channels related to
    segmentation, including a primary channel for cytoplasm or nuclei and an
    optional secondary channel. It also sets appropriate tooltips for guidance.

    Returns:
        A tuple containing:
            - A list of QComboBox objects for channel selection.
            - A list of QLabel objects with associated descriptions for the combo boxes.
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
    ModelButton is a class that represents a button for triggering segmentation computations based on a specified model.

    Attributes:
    - model_name: The name of the model associated with the button.
    - text: The display text of the button.

    Methods:
    - __init__
    - press

    This class is designed to allow users to initiate segmentation processes by pressing the button, with its behavior influenced by the model name and the text it displays. Upon initialization, it configures essential properties, ensuring proper integration with its parent context, while pressing triggers the associated computation.
    """

    def __init__(self, parent, model_name, text):
        """
        Initializes the TrainHelpWindow, setting up the user interface for displaying training instructions.

        This constructor sets the geometry and title of the window, creates a layout for the content, and loads
        the training instructions from an HTML file for display.

        Args:
            parent: The parent widget of this window. It can be None.

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

        Args:
            parent: The parent object that has the compute_segmentation method.

        Returns:
            None
        """
        parent.compute_segmentation(model_name=self.model_name)


class DenoiseButton(QPushButton):
    """No valid docstring found."""

    def __init__(self, parent, text):
        """
        Initializes a HelpWindow instance, setting up the user interface.

        This method creates a new HelpWindow, sets its geometry and title,
        loads help text from an HTML file, and displays it within the window.

        Args:
            parent: The parent widget of the HelpWindow. If no parent is provided,
                    the window will be a main window.

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
        Presses the segmentation computation on the specified parent object.

        Args:
            parent: The parent object that will perform the segmentation computation using the model.

        Returns:
            None: This method does not return a value.
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
    TrainWindow is a dialog for configuring training parameters in a machine learning pipeline.

    Attributes:
    - model_strings: A list of model names available for selection by the user.
    - training_files: A list of currently selected training files for model training.
    - current_channel: The channel selected for the training process.
    - training_parameters: A dictionary containing user-defined training parameters.

    This class provides a user interface for setting up various aspects of the training process, allowing users to select models, input training files, and specify other relevant training settings.
    """

    def __init__(self, parent, model_strings):
        """
        Initializes the ExampleGUI class, setting up the main window layout and loading an image.

        Args:
            parent: The parent widget for this GUI. Defaults to None if no parent is specified.

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
        Sets the training parameters for the model based on user input.

        Args:
            parent: An object that will receive the training parameters.

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
    A class that represents a graphical user interface for demonstrating functionality.

    Class Methods:
    - __init__:
    """

    def __init__(self, parent=None):
        """
        Initializes the training settings dialog for model training.

        This method sets up the graphical user interface for configuring various
        training parameters, including model selection, channel selection, learning
        rate, weight decay, the number of epochs, and the option to use SGD or AdamW.
        It also lists the files available in the current folder and their corresponding
        mask counts.

        Args:
            parent: The parent widget that this dialog is associated with.
            model_strings: A list of model names to populate the model selection combo box.

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
    """No valid docstring found."""

    def __init__(self, parent=None):
        """


        Initializes an instance of the class and configures its properties.



        This method sets up the instance by disabling it, setting its model type and text,

        configuring its font, and connecting a click event to a specified method.



        Args:

            parent: The parent object that provides context and resources for this instance.

            text: The text to be displayed on the instance.



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
    Class representing a help window for training.

    This class creates a window that displays helpful information related to training. It is designed to provide users with guidance and support while interacting with training functionalities.

    Attributes:
    - parent: The parent widget of this window.
    - title: The title of the help window.
    - content: The content to be displayed in the help window.
    - layout: The layout for organizing the widgets in the window.

    Methods:
    - __init__: Initializes the TrainHelpWindow instance.
    - show: Displays the help window to the user.

    The `__init__` method sets up the initial state of the window, including its geometry and layout. The `show` method makes the help window visible to the user.
    """

    def __init__(self, parent=None):
        """
        Initializes the class, setting up the user interface element and connecting signals.

        Args:
            parent: The parent widget that holds this interface element.
            model_name: The name of the model, which may be modified if it contains "cyto3".
            text: The text to display on the interface element.

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
    """No valid docstring found."""

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
        No valid docstring found.
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
        Initializes a new instance of the class, configuring the view box with optional parameters.

        Args:
            parent: The parent widget or object to which this instance belongs.
            border: The border configuration for the view box.
            lockAspect: A boolean flag indicating whether to lock the aspect ratio.
            enableMouse: A boolean flag indicating whether mouse interaction is enabled.
            invertY: A boolean flag to invert the Y-axis.
            enableMenu: A boolean flag to enable the menu functionality.
            name: The name of the view box instance.
            invertX: A boolean flag to invert the X-axis.

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
        Handles mouse click events for the application, managing different interactions based on the current state and button pressed.

        Parameters:
            ev: An event object representing the mouse click, which includes details such as the button pressed and its position.

        Returns:
            None: This method does not return a value.
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
        Handles mouse drag events by ignoring the given event.

        Args:
            ev: The event object that contains information about the mouse drag event.

        Returns:
            None: This method does not return a value.
        """
        ev.ignore()
        return

    def hoverEvent(self, ev):
        """
        Handles hover events for drawing strokes in the application. It processes the event based on the current stroke state and updates the drawing accordingly.

        Args:
            ev: The event object containing information about the hover event, including the position of the cursor.

        Returns:
            None: This method does not return any value.
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
        Creates a scatter plot item at the specified position and adds it to the parent's plot.

        Args:
            pos: The position at which the scatter plot item is created.

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
        Determines if the current stroke has returned to the starting point.

        This method checks the current stroke recorded in the parent object and evaluates
        if the stroke has returned to the starting position after moving away from it.
        It utilizes distance thresholds to determine if the stroke has left the start
        and subsequently returned.

        Args:
            pos: The current position in the stroke to be evaluated.

        Returns:
            bool: True if the stroke has returned to the starting point, False otherwise.
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
        Ends the current stroke by removing the graphical representation, appending it to the stroke list if necessary, and performing autosave operations.

        This method updates the state of the parent to indicate that the current stroke has ended. It checks if the current stroke should be appended to the list of strokes, processes the current stroke data, and manages the current point set. Additionally, it handles autosave functionality if enabled.

        Parameters:
        - self: The instance of the class containing the method.

        Returns:
        - None: This method does not return a value.
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

        This method processes tablet events and can be used to extract information
        from the event, such as the device, pointer type, and pressure. The actual
        implementation for handling the event needs to be defined.

        Args:
            ev: The event object that contains information about the tablet event.

        Returns:
            None: This method does not return a value.
        """
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        """
        Draws a graphical representation at the specified position on an image.

        This method uses a stroke mask and the current stroke from the parent context to draw at the given position. It calculates the appropriate slices for the mask and image based on the position and the dimensions of the drawing kernel.

        Args:
            pos: The position at which to draw, provided as a 2D coordinate.
            ev: An optional event parameter that may influence the drawing behavior.

        Returns:
            None
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
        Sets the draw kernel for the object.

        This method initializes a square kernel for drawing operations with a specified size.
        It creates a binary mask and an opacity mask based on the kernel, which are used for rendering.

        Args:
            kernel_size: The size of the kernel. It defines the dimensions of the square kernel,
                         which will be (kernel_size x kernel_size).

        Returns:
            None: This method does not return any value, but modifies the object's draw kernels
            and masks for drawing operations.
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
