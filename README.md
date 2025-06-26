# cellpose_plus

---

[![PyPi](https://badge.fury.io/py/cellpose_plus.svg)](https://badge.fury.io/py/cellpose_plus)
![License](https://img.shields.io/github/license/ITMO-MMRM-lab/cellpose_plus?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

Cellpose plus is a user-friendly morphological analysis tool that enhances the feature extraction of stained cell images. It streamlines the workflow from raw image input to insightful metrics, making it accessible for users to analyze and understand cellular morphology efficiently.

---

## Table of Contents

- [Core features](#core-features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---
## Core features

1. **Morphological Analysis Tool**: Cellpose plus serves as a morphological analysis tool specifically designed for feature extraction from stained cell images, enhancing the capabilities of the base Cellpose framework.
2. **Feature Extraction Algorithms**: The project includes advanced algorithms for extracting morphological properties such as area, roundness, size ratios, and more from cell images, providing comprehensive insights into cell and nucleus morphology.
3. **Single Workflow Integration**: It integrates a streamlined workflow that takes users from raw image inputs through segmentation to the generation of metrics and results, simplifying the overall analysis process for end-users.

---

## Installation

Install cellpose_plus using one of the following methods:

**Using PyPi:**

```sh
pip install cellpose_plus
```

To install Cellpose plus, you can use either `conda` or `pip`. Make sure you have Python 3.8 or higher.

1. Install [Anaconda](https://www.anaconda.com/download/).
2. Open a command prompt or Anaconda prompt.
3. Create a new environment for CPU only:
```bash
   conda create -n cellpose_plus 'python==3.9' pytorch
```
4. Activate the environment:
```bash
   conda activate cellpose_plus
```
5. For NVIDIA GPUs, use:
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
6. Install Cellpose plus and its dependencies:
```bash
   pip install cellpose-plus[gui]
```
   or for a version without GUI:
```bash
   pip install cellpose-plus
```
7. Optionally, you can install additional dependencies using:
```bash
   pip install -r /cellpose_plus/requirements.txt
```
## Getting Started

To get started with Cellpose plus, follow these simple steps to use the tool for morphological analysis of stained cell images:

### Running Cellpose plus
To launch the GUI and start using the tool:
1. Open a command line terminal/Anaconda Prompt.
2. Activate the respective environment:
```bash
   conda activate cellpose_plus
```
3. Start the GUI by entering:
```bash
   python -m cellpose
```
4. You can then load or drag-drop your desired images for segmentation.

   ![demo_gif](https://raw.githubusercontent.com/ITMO-MMRM-lab/cellpose/refs/heads/main/repo/demo_cellpose_plus.gif)

### Important Reminder
Before you calculate metrics, it’s mandatory to set a pixel-to-micrometer (μm) conversion value in the GUI. This is done by entering the value in the field labeled “Length in μm.” If a corresponding metadata file is present, it will automatically acquire this value.

After segmentation, you can save the results as masks and export the metrics to a folder. The outputs will be saved in CSV format and include detailed measurements for segmented cells.

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/docs/notebook.rst).

---

## Documentation

A detailed cellpose_plus description is available [here](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/ITMO-MMRM-lab/cellpose_plus/issues)**: Submit bugs found or log feature requests for the project.

---

## License

This project is protected under the BSD 3-Clause "New" or "Revised" License. For more details, refer to the [LICENSE](https://github.com/ITMO-MMRM-lab/cellpose_plus/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-MMRM-lab (2023). cellpose_plus repository [Computer software]. https://github.com/ITMO-MMRM-lab/cellpose_plus

### BibTeX format:

    @misc{cellpose_plus,

        author = {ITMO-MMRM-lab},

        title = {cellpose_plus repository},

        year = {2023},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-MMRM-lab/cellpose_plus.git}},

        url = {https://github.com/ITMO-MMRM-lab/cellpose_plus.git}

    }

---
