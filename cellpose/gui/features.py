import sys, os, pathlib, warnings, datetime, time, copy, math
from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu

from . import symmetry
from ..utils import download_font
import pandas as pd
import numpy as np
from scipy.stats import mode
import cv2

from scipy.ndimage import find_objects
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
from scipy import ndimage
import diplib as dip
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB = True
except:
    MATPLOTLIB = False


class FeatureExtraction:
    """
    A class for extracting features from images and managing mask operations.

    This class provides functionalities for feature extraction, including
    scaling contours, generating masks, creating colormaps, and calculating
    various image metrics. It serves as a core component for image analysis
    in contexts such as biological imaging.

    Methods:
        __init__
        save_temp_output
        scale_contour
        save_labeled_masks
        create_colormap_mask
        image_labeling
        mask_indexing
        create_colormap
        find_overlap
        matched_indices
        get_metrics
        out_concat
        get_voronoi_entropy
        save_metrics
        calculate_metrics
        select_mask
        select_image

    Attributes:
        main_masks_menu
        main_images_menu
        indexCytoMask
        indexNucleusMask
        currentImageMask
        current_model
        temp_masks

    - main_masks_menu: A pointer to the masks menu used for managing mask types.
    - main_images_menu: A pointer to the images menu for displaying images.
    - indexCytoMask: Index for the cytoplasmic mask utilized during analyses.
    - indexNucleusMask: Index for the nucleus mask, crucial for feature extraction.
    - currentImageMask: The currently active image mask being processed.
    - current_model: Indicates the name of the model currently in use for extraction tasks.
    - temp_masks: A list designated to hold temporary masks generated during operations.

    The class encompasses methods for saving outputs, scaling contours, labeling images,
    generating colormaps, and calculating structural metrics from binary masks.
    """

    def __init__(self):
        """
        Initializes a FeatureExtraction instance.

            This method sets up the initial state of the FeatureExtraction object by
            initializing various attributes and calling a function to download font
            resources if the graphical user interface (GUI) is not available.

            Attributes:
                main_masks_menu: A pointer to the masks menu.
                main_images_menu: A pointer to the images menu.
                indexCytoMask: Index for the cytoplasmic mask.
                indexNucleusMask: Index for the nucleus mask.
                currentImageMask: The current image mask being processed.
                current_model: The name of the current model in use.
                temp_masks: A list to hold temporary masks.

            Returns:
                None
        """
        super(FeatureExtraction, self).__init__()

        self.main_masks_menu = None  # Pointer to masks menu
        self.main_images_menu = None  # Pointer to images menu

        self.indexCytoMask = -1
        self.indexNucleusMask = -1
        self.currentImageMask = ""
        self.current_model = ""
        self.temp_masks = []

        download_font()  # If running without GUI

    def save_temp_output(self, masks="", image="", model_name="", gui_self=""):
        """
        Saves temporary masks and images based on user input and model specifications.

            This method handles the creation and storage of temporary mask and image outputs.
            If an image is not provided, it generates a new mask based on the current model or the specified model name.
            It also creates actions for selecting these masks as primary or secondary. If an image is provided,
            it adds the image to the main images menu for selection.

            Args:
                masks: A string representing masks, if any, to be processed.
                image: A string representing the image to be processed or stored.
                model_name: A string for the model name; if empty, the current model is used.
                gui_self: The GUI interface object used for creating UI elements and logging.

            Returns:
                None: This method does not return a value but logs the storage of the mask temporarily.
        """
        d = datetime.datetime.now()
        temp_output_name = gui_self.current_model if model_name == "" else model_name

        if image == "":
            mask_names = [
                mask_name[0]
                for mask_name in self.temp_masks
                if temp_output_name in mask_name[0]
                and mask_name[0][len(temp_output_name)] == "_"
            ]
            new_mask_names = temp_output_name + "_" + str(len(mask_names) + 1)
            subMenu = self.main_masks_menu.addMenu("&" + new_mask_names)

            cytoAction = QAction("Select as main mask (cytoplasm)", gui_self)
            cytoAction.triggered.connect(
                lambda checked, subMenu=subMenu, curr_index=len(
                    self.temp_masks
                ): self.select_mask(subMenu, "primary", curr_index, gui_self)
            )

            nucleiAction = QAction("Select as secondary mask (nucleus)", gui_self)
            nucleiAction.triggered.connect(
                lambda checked, subMenu=subMenu, curr_index=len(
                    self.temp_masks
                ): self.select_mask(subMenu, "secondary", curr_index, gui_self)
            )

            subMenu.addAction(cytoAction)
            subMenu.addAction(nucleiAction)

            self.temp_masks.append((new_mask_names, gui_self.cellpix[-1]))  # masks[-1]
        else:  # elif masks == "":
            if self.indexCytoMask > -1:
                full_name = (
                    temp_output_name + " " + self.temp_masks[self.indexCytoMask][0]
                )
                newImage = QAction(full_name, gui_self)
                newImage.triggered.connect(
                    lambda checked, image=image, name=full_name: self.select_image(
                        gui_self, image, name
                    )
                )
                self.main_images_menu.addAction(newImage)

        gui_self.logger.info(str(temp_output_name) + " mask stored temporarily")

    def scale_contour(self, cnt, scale):
        """
        Scales a contour around its centroid by a given scale factor.

            This method calculates the centroid of the provided contour, normalizes
            the contour points, scales them by the specified factor, and then
            translates them back to the original centroid position.

            Args:
                cnt: The contour to be scaled, represented as a numpy array of shape (n, 2),
                      where n is the number of points in the contour.
                scale: A scaling factor that determines how much to scale the contour.
                       A value greater than 1 enlarges the contour, while a value
                       between 0 and 1 shrinks it.

            Returns:
                A numpy array of scaled contour points, represented as a numpy array
                of the same shape (n, 2), with points adjusted according to the
                specified scale factor.
        """
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    def save_labeled_masks(self, gui_self):
        """save masks to *_mask.jpg"""

        # Create results dir
        results_dir = os.path.splitext(gui_self.filename)[0]
        labels_dir = results_dir + "/labels"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

        slices = find_objects(gui_self.cellpix[0].astype(int))
        for idx in range(gui_self.cellpix[0].max()):
            tmp_cellpix = np.copy(gui_self.cellpix[0])
            tmp_cellpix[idx + 1 != gui_self.cellpix[0]] = 0
            tmp_cellpix[idx + 1 == gui_self.cellpix[0]] = 255

            mask = tmp_cellpix.astype(np.uint8)

            im = Image.fromarray(mask)
            label_name = labels_dir + "/" + str(idx + 1) + ".png"
            im.save(label_name)

        tmp_cellpix = np.copy(gui_self.cellpix[0])
        new_cellpix = np.zeros_like(tmp_cellpix)
        for idx in range(gui_self.cellpix[0].max()):
            tmp_mask = np.copy(gui_self.cellpix[0])
            tmp_mask[idx + 1 != gui_self.cellpix[0]] = 0
            tmp_mask[idx + 1 == gui_self.cellpix[0]] = 255

            contours, _ = cv2.findContours(
                tmp_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [self.scale_contour(contour, 0.95) for contour in contours]
            new_contours = cv2.drawContours(
                new_cellpix, contours, -1, (255, 255, 255), -1
            )
            # new_cellpix[self.scale_contour(contours[0], 0.9) == 255] = 255
            # new_cellpix[tmp_mask == 255] = 255

        mask_tmp = new_cellpix.astype(np.uint8)

        im_tmp = Image.fromarray(mask_tmp)
        label_name = labels_dir + "/" + "total_mask.png"
        im_tmp.save(label_name)

    def create_colormap_mask(self, mask):
        """
        Generate a colormap mask from the provided mask.

            This method creates an RGBA image where the colors are determined by a generated colormap
            based on the input mask values. The mask's values are used to index into the colormap,
            resulting in a colored representation while applying transparency based on the mask.

            Args:
                mask: An array-like structure representing the mask, which contains indices that will
                      be used to reference the generated colormap.

            Returns:
                An array representing the RGBA image with color mapped from the provided mask,
                where the transparency is adjusted based on the mask values.
        """
        colormap = ((np.random.rand(1000000, 3) * 0.8 + 0.1) * 255).astype(np.uint8)
        tmp_mask = np.copy(mask).astype(np.uint8)

        colors = colormap[: tmp_mask.max(), :3]
        cellcolors = np.concatenate(
            (np.array([[255, 255, 255]]), colors), axis=0
        ).astype(np.uint8)

        layerz = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)

        new_tmp_mask = tmp_mask[np.newaxis, ...]

        layerz[..., :3] = cellcolors[new_tmp_mask[0], :]
        layerz[..., 3] = 128 * (new_tmp_mask[0] > 0).astype(np.uint8)

        return layerz

    def image_labeling(self, im_mask="", im_labels="", coords=""):
        """
        Label the regions in an image mask with specified labels and coordinates.

            This method takes an image mask, optional labels, and coordinates to draw
            text labels onto the image mask at the specified coordinates. If no labels
            are provided, it defaults to using consecutive integers as labels.

            Args:
                im_mask: The image mask to be labeled.
                im_labels: Optional labels for each region in the image mask. If not
                           provided, consecutive integers will be used as labels.
                coords: The coordinates where the labels should be placed on the image mask.

            Returns:
                The labeled image mask with text drawn at the specified coordinates.
        """
        im_mask_labeled = im_mask.copy()

        font_path = pathlib.Path.home().joinpath(".cellpose", "DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), size=20)

        I1 = ImageDraw.Draw(im_mask_labeled)

        for idx in range(0, len(coords)):
            if im_labels == "":
                label_value = str(idx + 1)
            else:
                label_value = str(im_labels[idx])

            I1.text(
                (coords[idx][0], coords[idx][1]),
                label_value,
                anchor="mb",
                fill=(255, 255, 255),
                font=font,
            )

        return im_mask_labeled

    def mask_indexing(self, im_mask, coords):
        """
        Add indexed labels to coordinates on an image mask.

            This method modifies the given image mask by overlaying
            text labels at specified coordinates, allowing for easy
            identification of areas in the mask. The labels will start
            from 1 and increment for each coordinate provided.

            Args:
                im_mask: The image mask on which to overlay the labels.
                coords: A list of coordinate tuples where labels will be
                        placed on the image mask.

            Returns:
                A modified image mask with labeled coordinates.
        """
        im_mask_labeled = im_mask.copy()

        font_path = pathlib.Path.home().joinpath(".cellpose", "DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), size=20)

        I1 = ImageDraw.Draw(im_mask_labeled)

        for idx in range(0, len(coords)):
            I1.text(
                (coords[idx][0], coords[idx][1]),
                str(idx + 1),
                anchor="mb",
                fill=(255, 255, 255),
                font=font,
            )

        return im_mask_labeled

    def create_colormap(mask_cyto, mask_nuclei):
        """
        Generates color maps for cellular components from provided masks.

            This method creates color-mapped images for cytoplasm and nuclei based on the given masks.
            It also generates an overlap image where both cytoplasm and nuclei are present.

            Args:
                mask_cyto: The mask representing the cytoplasm regions.
                mask_nuclei: The mask representing the nuclei regions.

            Returns:
                A tuple containing three images: the cytoplasm color map, the nuclei color map,
                and the overlap color map indicating areas where both components are present.
        """
        # Cyto colormap
        layerz_cyto = create_colormap_mask(mask_cyto)
        im_cyto = Image.fromarray(layerz_cyto)

        # Nuclei colormap
        layerz_nuclei = create_colormap_mask(mask_nuclei)
        im_nuclei = Image.fromarray(layerz_nuclei)

        # Overlap colormap
        layerz_overlap = np.copy(layerz_cyto).astype(np.uint8)
        for idxi in range(0, layerz_overlap.shape[0]):
            for idxj in range(0, layerz_overlap.shape[1]):
                if (layerz_cyto[idxi][idxj] != [255, 255, 255, 0]).all() and (
                    layerz_nuclei[idxi][idxj] != [255, 255, 255, 0]
                ).all():
                    layerz_overlap[idxi][idxj] = [255, 0, 0, 128]
        im_overlap = Image.fromarray(layerz_overlap)

        return im_cyto, im_nuclei, im_overlap

    def find_overlap(self, cyto_mask, nuclei_mask, cyto_nuclei_indices):
        """
        Find the number of overlapping pixels between cellular and nuclei masks.

            This method counts the number of pixels in the cytoplasmic mask
            that correspond to specified indices in both the cytoplasmic mask
            and nuclei mask. It iterates over each pixel and increments the
            count whenever a pixel matches the specified indices.

            Args:
                cyto_mask: The mask representing cytoplasmic regions.
                nuclei_mask: The mask representing nuclei regions.
                cyto_nuclei_indices: A list or tuple containing two indices
                                     that specify the values to look for in
                                     the cyto_mask and nuclei_mask.

            Returns:
                The count of overlapping pixels between the cytoplasmic and
                nuclei masks that match the specified indices.
        """
        count = 0
        for idxi in range(0, cyto_mask.shape[0]):
            for idxj in range(0, cyto_mask.shape[1]):
                if (
                    cyto_mask[idxi][idxj] == cyto_nuclei_indices[0]
                    and nuclei_mask[idxi][idxj] == cyto_nuclei_indices[1]
                ):
                    count += 1
        return count

    def matched_indices(
        self, cyto_mask, nuclei_mask, cyto_size, nuclei_size, main_coords
    ):
        """
        Identify and match indices of cellular and nuclear regions based on provided masks
        and their corresponding sizes, while ensuring unique overlaps.

        Args:
            cyto_mask: A mask array representing cellular regions.
            nuclei_mask: A mask array representing nuclear regions.
            cyto_size: A list containing sizes of the cellular regions.
            nuclei_size: A list containing sizes of the nuclear regions.
            main_coords: A list of coordinates associated with the main features.

        Returns:
            A tuple containing:
                - cyto_nuclei_ratio: A list of ratios of cellular to nuclear sizes, rounded to two decimal places.
                - tmp_coords: A list of tuples representing the coordinates of the matched indices.
                - indices_cyto_nuclei: A list of unique matched indices of cellular and nuclear regions.
        """
        tmp_cyto = np.copy(cyto_mask)  # .astype(np.uint8)
        tmp_nuclei = np.copy(nuclei_mask)  # .astype(np.uint8)
        tmp_coords = np.copy(main_coords)

        indices_cyto_nuclei = set()

        # Remove duplicates
        for idxi in range(0, tmp_cyto.shape[0]):
            for idxj in range(0, tmp_cyto.shape[1]):
                if tmp_cyto[idxi][idxj] != 0 and tmp_nuclei[idxi][idxj] != 0:
                    indices_cyto_nuclei.add(
                        (tmp_cyto[idxi][idxj], tmp_nuclei[idxi][idxj])
                    )

        indices_cyto_nuclei = list(indices_cyto_nuclei)
        indices_cyto_nuclei.sort()

        # Assure 1 cyto for 1 nuclei
        n = len(indices_cyto_nuclei)
        cnt = 0

        while cnt < n - 1:
            if indices_cyto_nuclei[cnt][0] == indices_cyto_nuclei[cnt + 1][0]:
                to_del = (
                    self.find_overlap(tmp_cyto, tmp_nuclei, indices_cyto_nuclei[cnt])
                    > self.find_overlap(
                        tmp_cyto, tmp_nuclei, indices_cyto_nuclei[cnt + 1]
                    )
                ) * 1
                del indices_cyto_nuclei[cnt + to_del]
                n = n - 1
            else:
                cnt = cnt + 1

        cyto_nuclei_ratio = [
            round(
                cyto_size[index_cyto_nuclei[0] - 1]
                / nuclei_size[index_cyto_nuclei[1] - 1],
                2,
            )
            for index_cyto_nuclei in indices_cyto_nuclei
        ]
        tmp_coords = [
            tuple(tmp_coords[index_cyto_nuclei[0] - 1])
            for index_cyto_nuclei in indices_cyto_nuclei
        ]
        return cyto_nuclei_ratio, tmp_coords, indices_cyto_nuclei

    def get_metrics(self, mask, custom_features, gui_self):
        """
        Calculate size and roundness metrics of labeled regions in a binary mask.

            This method processes a binary mask to identify labeled regions,
            computes their center coordinates, sizes, and roundness based on
            provided custom features, and returns the calculated metrics.

            Args:
                mask: A binary mask array where labeled regions are encoded.
                custom_features: A set of custom features to compute for each labeled region.
                gui_self: An object providing GUI-related settings (e.g., options to calculate size and roundness).

            Returns:
                A tuple containing:
                    - A list of lists with size and roundness metrics for each labeled region.
                    - A list of center coordinates for each labeled region.
        """
        slices = ndimage.find_objects(mask.astype(int))
        center_coords = []
        size_cells = []
        round_cells = []

        for idx, si in enumerate(slices):
            mask_tmp = np.copy(mask).astype(np.uint8)
            mask_tmp[(idx + 1) != mask] = 0
            mask_tmp[(idx + 1) == mask] = 255

            padded_mask = np.pad(mask_tmp, 1, mode="constant")

            ####
            Zlabeled, Nlabels = ndimage.label(padded_mask)
            label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]

            # Remove the labels with size < 5
            for label, size in enumerate(label_size):
                if size < 5:
                    padded_mask[Zlabeled == label] = 0
            ####

            labels = dip.Label(padded_mask > 0)
            msr = dip.MeasurementTool.Measure(labels, features=custom_features)
            center_coords.append(
                [round(msr[1]["Center"][0], 2), round(msr[1]["Center"][1], 2)]
            )
            if gui_self.calcSize:
                size_cells.append(
                    round(msr[1]["SolidArea"][0] * pow(gui_self.px_to_mm, 2), 2)
                )
            if gui_self.calcRound:
                round_cells.append(round(msr[1]["Roundness"][0], 2))

        return [size_cells, round_cells], center_coords

    def out_concat(self, prev_out, curr_out):
        """
        Concatenates output values based on the type of the previous output.

            This method takes in a previous output and a current output,
            returning a list that either combines the previous output with
            the current output or reformats the output based on its type.

            Args:
                prev_out: The previous output value, which can either be a
                    float or a list.
                curr_out: The current output value, which is expected to
                    be compatible for concatenation with the previous output.

            Returns:
                A list containing the elements of the concatenated output.
                If the previous output is a float, the list contains the
                previous and current outputs. If the previous output is a
                list, the list contains the first two elements of the
                previous output and the current output.
        """
        if isinstance(prev_out, float):
            return [prev_out, curr_out]
        else:  # isinstance(prev_out, list)
            return [prev_out[0], prev_out[1], curr_out]

    def get_voronoi_entropy(self, vor):
        """
        Calculate the entropy of a Voronoi diagram based on the distribution of bounded polygon classes.

            This method computes the entropy of a Voronoi diagram by analyzing the number of bounded regions
            and their respective class proportions. It utilizes the regions defined in the Voronoi object to
            determine how many unique polygon classes are present and their contribution to the overall entropy.

            Args:
                vor: The Voronoi diagram object containing information about the regions and points.

            Returns:
                A float representing the calculated Voronoi entropy, rounded to three decimal places.
        """
        polygon_class_counts = {}

        for region_index in vor.point_region:
            region_vertices = vor.regions[region_index]

            # Exclude unbounded regions
            if -1 not in region_vertices:
                polygon_class = len(region_vertices)

                if polygon_class in polygon_class_counts:
                    polygon_class_counts[polygon_class] += 1
                else:
                    polygon_class_counts[polygon_class] = 1

        # Total number of bounded regions and proportions
        total_bounded_regions = sum(polygon_class_counts.values())
        proportions = {
            polygon_class: count / total_bounded_regions
            for polygon_class, count in polygon_class_counts.items()
        }

        # Voronoi entropy
        voronoi_entropy = -sum(p * math.log(p) for p in proportions.values() if p > 0)
        return round(voronoi_entropy, 3)

    def save_metrics(
        self,
        masks_img,
        center_coords,
        metric_cells,
        metric_name,
        out_csv,
        out_name,
        out_dir,
        gui_self,
    ):
        """
        Saves computed metrics and corresponding images to specified output.

            This method processes mask images and metric information to generate
            labeled images and a CSV file containing various metrics. The generated
            images include a colormap mask and a size-labeled image, and the
            metrics are saved in a CSV format based on the specified metric name.

            Args:
                masks_img: The image masks used for generating the colormap.
                center_coords: The coordinates of the centers for metrics labeling.
                metric_cells: The computed metrics for the images.
                metric_name: The name to use for the metric when saving the image.
                out_csv: A list of existing CSV data or an empty list to initiate.
                out_name: The name of the metric for the CSV output.
                out_dir: The directory where the images and CSV will be saved.
                gui_self: An object containing GUI-related data and functions.

            Returns:
                The updated list of metrics saved in CSV format.
        """
        layerz_cell = self.create_colormap_mask(masks_img)
        im_cell = Image.fromarray(layerz_cell)

        # Colormap img
        colormap_mask = Image.fromarray(self.create_colormap_mask(masks_img))
        im_masks = self.mask_indexing(colormap_mask, center_coords)
        im_masks.save(out_dir + "/" + "mask_colormap.png")

        # Metric img
        im_cell_size_labeled = self.image_labeling(
            im_mask=im_cell, im_labels=metric_cells, coords=center_coords
        )
        self.save_temp_output(
            image=im_cell_size_labeled, model_name=metric_name, gui_self=gui_self
        )
        im_cell_size_labeled.save(out_dir + "/" + metric_name + ".png")
        out_csv = (
            metric_cells
            if len(out_csv) == 0
            else [
                self.out_concat(out_csv[idx], single_metric)
                for idx, single_metric in enumerate(metric_cells)
            ]
        )

        # Metric csv
        header_title = []
        if out_name == "size":
            header_title = "area"
        elif out_name == "size_roundness":
            header_title = "area,roundness"
        elif out_name == "roundness":
            header_title = "roundness"
        elif out_name == "Center":
            header_title = "x,y"
        elif out_name == "ratio":
            header_title = "cell_id,nuclei_id,cell_nuclei_ratio"
        np.savetxt(
            out_dir
            + "/"
            + gui_self.filename.split("/")[-1].split(".")[0]
            + "_"
            + out_name
            + ".csv",
            out_csv,
            delimiter=", ",
            header=header_title,
            comments="",
            fmt="% s",
        )

        return out_csv

    def calculate_metrics(self, gui_self):
        """
        Calculate various metrics for the primary and secondary masks.

            This method evaluates metrics from main and secondary mask images, such as size and roundness,
            and generates output files, including Voronoi diagrams and convex hulls, in specified directories.

            Args:
                gui_self: An object containing parameters for calculation, including options to compute size,
                           roundness, ratios, and Voronoi metrics and the filename for saving results.

            Returns:
                tuple: A tuple containing the size metrics of the main mask and the coordinates of the centers.
        """
        main_masks_img = self.temp_masks[self.indexCytoMask][
            1
        ]  # self.temp_masks[-1][1]
        secondary_masks_img = self.temp_masks[self.indexNucleusMask][1]
        comparison = main_masks_img == secondary_masks_img
        comparison = comparison.all()
        print("Are they equal?: ", comparison)

        # Create results dir
        results_dir = os.path.splitext(gui_self.filename)[0]
        primary_results_dir = results_dir + "/primary"
        secondary_results_dir = results_dir + "/secondary"

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        if not os.path.exists(primary_results_dir):
            os.makedirs(primary_results_dir)
        if not os.path.exists(secondary_results_dir):  # and not comparison:
            os.makedirs(secondary_results_dir)

        output_csv_primary = []
        output_csv_secondary = []
        output_csv_ratio = []
        output_csv_coords = []
        output_csv_voronoi = []

        output_name = ""

        # Set dip metrics
        custom_features = ["Center"]
        if gui_self.calcSize:
            custom_features.append("SolidArea")
            output_name += "size"
        if gui_self.calcRound:
            custom_features.append("Roundness")
            output_name += "_roundness"

        # Metrics for main mask
        main_metrics, center_coords_main = self.get_metrics(
            main_masks_img, custom_features, gui_self
        )
        size_cells_main, round_cells_main = main_metrics
        if gui_self.calcRatio:
            # Metrics for secondary mask (if exists)
            secondary_metrics, center_coords_secondary = self.get_metrics(
                secondary_masks_img, custom_features, gui_self
            )
            size_cells_secondary, round_cells_secondary = secondary_metrics
            ratio_cells, center_coords_ratio, indices_cyto_nuclei = (
                self.matched_indices(
                    main_masks_img,
                    secondary_masks_img,
                    size_cells_main,
                    size_cells_secondary,
                    center_coords_main,
                )
            )
            output_csv_ratio = [
                [index_cyto_nuclei[0], index_cyto_nuclei[1]]
                for index_cyto_nuclei in indices_cyto_nuclei
            ]

        for idx_feature, feature in enumerate(custom_features):
            if idx_feature > 0:
                output_csv_primary = self.save_metrics(
                    main_masks_img,
                    center_coords_main,
                    main_metrics[idx_feature - 1],
                    custom_features[idx_feature],
                    output_csv_primary,
                    output_name,
                    primary_results_dir,
                    gui_self,
                )

                if gui_self.calcRatio:
                    output_csv_secondary = self.save_metrics(
                        secondary_masks_img,
                        center_coords_secondary,
                        secondary_metrics[idx_feature - 1],
                        custom_features[idx_feature],
                        output_csv_secondary,
                        output_name,
                        secondary_results_dir,
                        gui_self,
                    )

        if gui_self.calcRatio:
            output_csv_ratio = self.save_metrics(
                main_masks_img,
                center_coords_ratio,
                ratio_cells,
                "ratio",
                output_csv_ratio,
                "ratio",
                results_dir,
                gui_self,
            )

        output_csv_coords = self.save_metrics(
            main_masks_img,
            center_coords_main,
            center_coords_main,
            "Center",
            output_csv_coords,
            "Center",
            primary_results_dir,
            gui_self,
        )

        if gui_self.calcVoronoi:
            img = plt.imread(gui_self.filename)

            output_csv_coords = pd.DataFrame(output_csv_coords)
            output_csv_coords[1] = img.shape[0] - output_csv_coords[1]

            vor = Voronoi(output_csv_coords)
            fig = voronoi_plot_2d(vor)

            fig, ax = plt.subplots()
            ax.imshow(ndimage.rotate(np.fliplr(img), 180))
            fig = voronoi_plot_2d(vor, point_size=10, ax=ax, line_colors="red")
            plt.savefig(results_dir + "/" + "voronoi.png")

            ### Convex Hull

            conhull = ConvexHull(output_csv_coords)
            fig = convex_hull_plot_2d(conhull)

            fig, ax = plt.subplots()
            ax.imshow(ndimage.rotate(np.fliplr(img), 180))
            fig = convex_hull_plot_2d(conhull, ax=ax)
            plt.savefig(results_dir + "/" + "hull.png")

            conhull_area = conhull.area
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_convex_hull_area.csv",
                [round(conhull_area, 3)],
                delimiter=", ",
                header="convex_hull_area",
                comments="",
                fmt="% s",
            )
            ###

            voronoi_entropy = self.get_voronoi_entropy(vor)
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_vornoi_entropy.csv",
                [voronoi_entropy],
                delimiter=", ",
                header="voronoi_entropy",
                comments="",
                fmt="% s",
            )

            CSM_array = symmetry.CSM_for_graph(vor)
            np.savetxt(
                results_dir
                + "/"
                + gui_self.filename.split("/")[-1].split(".")[0]
                + "_CSM_values.csv",
                [round(np.asarray(CSM_array).mean(), 3)],
                delimiter=", ",
                header="CSM_array",
                comments="",
                fmt="% s",
            )

        return size_cells_main, center_coords_main

    def select_mask(self, menu_output, cell_type, curr_index, gui_self):
        """
        Select and update the mask for a specified cell type.

            This method updates the current selection of masks based on the input parameters
            and modifies the icon of the selected mask in the provided menu output. It also enables
            or disables associated checkboxes in the GUI based on the current mask selection state.

            Args:
                menu_output: The menu output widget to update the mask icon.
                cell_type: The type of cell ('primary' or 'secondary') for which the mask is being selected.
                curr_index: The index of the currently selected mask in the menu.
                gui_self: The GUI object that contains checkboxes and buttons for enabling or disabling
                          based on the selected masks.

            Returns:
                None: This method does not return a value, but it alters the state of the GUI
                and the mask selections based on input parameters.
        """
        if cell_type == "primary":
            if self.indexCytoMask != -1:
                prev_selected_mask = menu_output.parentWidget().findChildren(QMenu)[
                    self.indexCytoMask
                ]
                prev_selected_mask.setIcon(QtGui.QIcon())
            self.indexCytoMask = curr_index
            self.indexNucleusMask = (
                -1 if self.indexNucleusMask == curr_index else self.indexNucleusMask
            )
        elif cell_type == "secondary":
            if self.indexNucleusMask != -1:
                prev_selected_mask = menu_output.parentWidget().findChildren(QMenu)[
                    self.indexNucleusMask
                ]
                prev_selected_mask.setIcon(QtGui.QIcon())
            self.indexNucleusMask = curr_index
            self.indexCytoMask = (
                -1 if self.indexCytoMask == curr_index else self.indexCytoMask
            )
        icon_path = pathlib.Path.home().joinpath(".cellpose", str(cell_type) + ".png")
        menu_output.setIcon(QtGui.QIcon(str(icon_path.resolve())))

        gui_self.RTCheckBox.setEnabled(
            self.indexCytoMask > -1 and self.indexNucleusMask > -1
        )
        gui_self.VDCheckBox.setEnabled(
            self.indexCytoMask > -1 and self.indexNucleusMask > -1
        )
        gui_self.SMCheckBox.setEnabled(self.indexCytoMask > -1)
        gui_self.RMCheckBox.setEnabled(self.indexCytoMask > -1)
        # self.CalculateButton.setStyleSheet(self.styleUnpressed if self.indexCytoMask > -1 else self.styleInactive)
        gui_self.CalculateButton.setEnabled(self.indexCytoMask > -1)

    def select_image(self, gui_self, img_layer, name):
        """
        Select and update the displayed image layer based on the provided name.

            This method checks if the currently selected image mask is different from
            the specified name. If it is, the method updates the displayed image layer
            with the new image data. If the name matches the current image mask, it
            updates the layer without changing the image.

            Args:
                gui_self: The GUI object that contains the layer being updated.
                img_layer: The image layer data to be displayed.
                name: The name associated with the image to select.

            Returns:
                None
        """
        if self.currentImageMask != name:
            gui_self.layer.setImage(np.asarray(img_layer), autoLevels=False)
            self.currentImageMask = name
        else:
            self.update_layer()
            self.currentImageMask = ""
        print("WEEEEEE 3")
