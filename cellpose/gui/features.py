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
    """No valid docstring found."""

    def __init__(self):
        """
        Initializes a FeatureExtraction object, setting up various attributes related to mask and image menus.

        This constructor method initializes several attributes used in the FeatureExtraction class, including pointers
        to the main masks and images menus, indices for cytoplasm and nucleus masks, the current image mask, the
        current model, and a temporary list for masks. It also ensures that the necessary font is downloaded if
        the application is running without a graphical user interface.

        Parameters:
        - None

        Returns:
        - None
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
        Saves temporary output masks or images and updates the respective GUI menus.

        This method handles the creation of temporary mask names if no existing masks are specified. It updates the main masks and images menus in the GUI, allowing the user to select the temporary masks as primary or secondary. It also logs the action of storing the mask temporarily.

        Args:
            masks: The masks to be processed. If not provided, a new mask will be created based on the current model or given model name.
            image: The image associated with the mask. If not provided, a new mask will be created.
            model_name: The name of the model to reference for naming the new mask. Defaults to the current model if not provided.
            gui_self: The reference to the GUI instance that is being used to update the menus and log actions.

        Returns:
            None
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
        Scales the given contour by a specified scale factor around its centroid.

        This method computes the centroid of the input contour, normalizes the contour points, scales them by the given factor, and then translates them back to their original position.

        Args:
            cnt: A 2D array of contour points.
            scale: A scaling factor to resize the contour.

        Returns:
            A 2D array of scaled contour points as integers.
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
        Creates a colormap mask from the provided mask.

        This method generates a random colormap and applies it to the input mask, creating a new array that contains color-coded layers with transparency based on the mask values.

        Args:
            mask: The input mask array from which the colormap will be generated.

        Returns:
            A 3D numpy array representing the colored mask with an alpha channel, where the color is determined by the input mask values and transparency is set according to the mask presence.
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
        Label an image mask with specified labels and coordinates.

        This method draws labels on an image mask at the specified coordinates.
        If no labels are provided, it automatically generates sequential labels.

        Args:
            im_mask: The image mask to be labeled.
            im_labels: The labels to use for labeling the image mask.
            coords: The coordinates where the labels should be drawn on the image.

        Returns:
            The labeled image mask with labels drawn at the specified coordinates.
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
        Masks an image and labels it with indexed coordinates.

        Args:
            im_mask: A copy of the input image mask to be labeled.
            coords: A list of coordinate tuples where text labels will be placed.

        Returns:
            An image mask with indexed labels drawn at specified coordinates.
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
        Creates color mappings for cellular components from given masks.

        This method generates three color-mapped images: one for cytoplasm, one for nuclei, and a combined overlay image showing where both components are present.

        Args:
            mask_cyto: A mask representing the cytoplasm regions.
            mask_nuclei: A mask representing the nuclei regions.

        Returns:
            A tuple containing:
                - im_cyto: A color-mapped image for the cytoplasm.
                - im_nuclei: A color-mapped image for the nuclei.
                - im_overlap: A color-mapped image showing the overlap between cytoplasm and nuclei.
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
        No valid docstring found.
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
        Calculates the matched indices between cytoplasm and nuclei masks.

        This method identifies unique pairs of cytoplasm and nuclei indices based on their presence in the provided masks.
        It ensures that each cytoplasm index is matched with at most one nuclei index by checking for overlaps.
        The method returns the cytoplasm to nuclei size ratios, the coordinates of the matched indices, and the sorted list of matched indices.

        Args:
            cyto_mask: A 2D array representing the cytoplasm mask.
            nuclei_mask: A 2D array representing the nuclei mask.
            cyto_size: A list containing sizes corresponding to each cytoplasm index.
            nuclei_size: A list containing sizes corresponding to each nuclei index.
            main_coords: A list of coordinates corresponding to the indices.

        Returns:
            A tuple containing:
                - A list of ratios of cytoplasm sizes to nuclei sizes for matched indices.
                - A list of coordinates for the matched cytoplasm and nuclei indices.
                - A list of matched indices as tuples.
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
        Calculates various metrics for labeled regions in a given mask.

        This method analyzes the input mask to extract metrics such as size and roundness of labeled regions,
        as well as their center coordinates. It applies a series of image processing techniques to identify and
        measure the specified features for each region.

        Args:
            mask: A binary or labeled mask from which metrics are to be extracted.
            custom_features: A list of features to be used for measuring the labeled regions.
            gui_self: An object containing GUI-related attributes for calculations, including flags for size and roundness.

        Returns:
            A tuple containing two elements:
                - A list of lists with sizes and roundness metrics for the identified regions.
                - A list of center coordinates for the identified regions.
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
        No valid docstring found.
        """
        if isinstance(prev_out, float):
            return [prev_out, curr_out]
        else:  # isinstance(prev_out, list)
            return [prev_out[0], prev_out[1], curr_out]

    def get_voronoi_entropy(self, vor):
        """
        Calculates the entropy of a Voronoi diagram based on the distribution of bounded regions.

        This method analyzes the Voronoi regions and computes the entropy, which quantifies the diversity of the shapes based on their proportions. It excludes unbounded regions from the calculation and returns the rounded entropy value.

        Args:
            vor: The Voronoi object that contains the regions and their corresponding vertices.

        Returns:
            A float representing the rounded Voronoi entropy value.
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
        Saves various metrics as images and a CSV file.

        This method generates and saves a colormap of masks, labeled images for metrics,
        and a CSV file containing the desired metrics based on the input parameters.

        Args:
            masks_img: The image containing masks to be processed and visualized.
            center_coords: Coordinates indicating the center points of masks.
            metric_cells: The metrics for the cell(s) to be saved.
            metric_name: The name to be used for the saved metric image file.
            out_csv: An existing CSV output data or an empty list to create a new one.
            out_name: The specific metric name to determine the CSV header.
            out_dir: The directory where the output images and CSV will be saved.
            gui_self: An instance of the GUI component for file handling and UI operations.

        Returns:
            The updated list of metrics saved into the CSV format.
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
        Calculates various metrics based on the provided mask images,
        compares them, and saves the results into designated directories.

        Args:
            gui_self: An object that contains options for calculations, including
                      filename, and flags for size, roundness, ratio, and Voronoi calculations.

        Returns:
            A tuple containing size metrics of the main mask and the center coordinates.
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
        Selects the appropriate mask based on the cell type and updates the GUI accordingly.

        This method modifies the state of the GUI by enabling or disabling checkboxes and updating icons
        based on the selected masks for primary and secondary cell types.

        Args:
            menu_output: The menu output widget that holds the mask options.
            cell_type: The type of cell for which the mask is being selected (either "primary" or "secondary").
            curr_index: The index of the currently selected mask.
            gui_self: The GUI instance that contains the checkboxes and button to be enabled or disabled.

        Returns:
            None: This method does not return any value.
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
        Selects and updates the image layer in the GUI.

        If the current image mask is different from the provided name, it sets the new image
        for the specified layer. If the current image mask is the same as the provided name,
        it updates the layer and clears the current image mask.

        Args:
            gui_self: The GUI instance that manages the image layers.
            img_layer: The image data to be displayed in the layer.
            name: The identifier name for the current image mask.

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
