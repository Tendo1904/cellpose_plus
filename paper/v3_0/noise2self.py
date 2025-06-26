"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import time
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import transforms, io, metrics
from cellpose.models import CellposeModel

# uses torch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torch.nn import MSELoss
from torch.optim import Adam

# PATH TO REPO
import sys

sys.path.append("/github/noise2self/")
from mask import Masker
from models.unet import Unet

device = torch.device("cuda")


class Cells(Dataset):
    """
    A class to handle a dataset of images, providing functionality to retrieve
    transformed images by index.

    Methods:
        __init__: Initializes the class with data and a specified image size.
        __len__: Returns the length of the data.
        __getitem__: Retrieves a transformed image associated with the given index.

    Attributes:
        data: The dataset or input data to be processed.
        xy: A tuple representing the desired height and width of the images after resizing.

    The class allows users to initialize it with input data and specify the
    dimensions for image resizing. It provides methods to access the length of
    the dataset and retrieve images in a transformed format suitable
    for further processing.
    """

    def __init__(self, data, xy=(128, 128)):
        """
        Initialize the class with data and a specified image size.

            This method sets up the initial state of the instance by storing the provided data
            and defining a series of transformations to be applied to the images in the dataset.

            Args:
                data: The dataset or input data to be processed.
                xy: A tuple representing the desired height and width of the images
                    after resizing (default is (128, 128)).

            Returns:
                None
        """
        self.data = data
        self.cell_transforms = v2.Compose(
            [
                v2.RandomRotation(degrees=180),
                v2.RandomResizedCrop(
                    size=xy, scale=(0.75, 1.25), ratio=(1.0, 1.0), antialias=True
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        """
        Returns the length of the data.

            This method provides the number of items in the data.

            Returns:
                The length of the data as an integer.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a transformed image associated with the given index.

            This method takes an index and returns the corresponding image after applying
            the necessary transformations. It ensures that the data is converted to a
            suitable format for further processing.

            Args:
                index: The index of the image to retrieve from the dataset.

            Returns:
                A tensor representing the transformed image.
        """
        img = self.cell_transforms(torch.from_numpy(self.data[index]).to(device))
        return img


def train_per_image(img_noisy):
    """
    Train a U-Net model for image denoising on a single noisy image.

        This method trains a U-Net model using the provided noisy image. The training involves
        creating batches of the noisy image, applying a mask, and optimizing the model with a
        mean squared error loss function. After training, it evaluates the model to produce a
        denoised version of the input image.

        Args:
            img_noisy: The input noisy image that will be used for training and denoising.

        Returns:
            The denoised output image as a NumPy array.
    """

    masker = Masker(width=4, mode="interpolate")
    torch.cuda.manual_seed(0)
    model = Unet().to(device)

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-4)

    cells_train = Cells(
        np.tile(img_noisy[np.newaxis, ...], (8, 1, 1, 1)), xy=(128, 128)
    )
    data_loader = DataLoader(cells_train, batch_size=8, shuffle=True)

    for ep in range(100):
        for k, batch in enumerate(data_loader):
            noisy_images = batch

            net_input, mask = masker.mask(noisy_images, k)
            net_output = model(net_input)

            loss = loss_function(net_output * mask, noisy_images * mask)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    img = img_noisy.copy()
    img, ysub, xsub = transforms.pad_image_ND(img)
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        simple_output = model(img)
    out = (
        simple_output.squeeze()[ysub[0] : ysub[-1] + 1, xsub[0] : xsub[-1] + 1]
        .cpu()
        .numpy()
    )

    return out


def train_per_image_synthetic(root, ctype="cyto2", plot=False, save=True):
    """
    Trains a segmentation model on synthetic noisy images for cell segmentation.

        This method processes a set of noisy images, applies a segmentation model, and optionally plots
        the results. The generated segmentations are stored in the provided root directory.

        Args:
            root: The root directory where the noisy test images are located and where results
                  will be saved if specified.
            ctype: The type of cell model to use for segmentation. Defaults to "cyto2".
            plot: A flag indicating whether to display the images and their corresponding masks.
                  Defaults to False.
            save: A flag indicating whether to save the output images and masks to a file.
                  Defaults to True.

        Returns:
            A tuple containing two lists:
            - A list of processed images after applying the segmentation.
            - A list of the corresponding segmentation masks generated by the model.
    """
    noise_type = "poisson"

    dat = np.load(
        root / "noisy_test" / f"test_{noise_type}.npy", allow_pickle=True
    ).item()
    test_noisy = dat["test_noisy"]
    test_labels = dat["masks_true"]
    diam_test = (
        dat["diam_test"] if "diam_test" in dat else 30.0 * np.ones(len(test_noisy))
    )

    imgs_n2s, masks_n2s = [], []

    seg_model = CellposeModel(gpu=True, model_type=f"{ctype}")
    for i in trange(len(test_noisy)):
        out = train_per_image(test_noisy[i])

        masks = seg_model.eval(
            out, diameter=diam_test[i], channels=[1, 0], channel_axis=0, normalize=True
        )[0]

        masks_n2s.append(masks)
        imgs_n2s.append(out)

        if plot:
            print(f">>> IMAGE {i}, n_masks = {masks.max()}")
            plt.figure(figsize=(12, 3))
            plt.subplot(1, 4, 1)
            plt.imshow(test_noisy[i][0])
            plt.subplot(1, 4, 2)
            plt.imshow(out)
            plt.subplot(1, 4, 3)
            plt.imshow(masks)
            plt.subplot(1, 4, 4)
            plt.imshow(test_labels[i])
            plt.show()

    dat["masks_n2s"] = masks_n2s
    dat["test_n2s"] = imgs_n2s
    if save:
        np.save(root / "noisy_test" / f"test_{noise_type}_n2s.npy", dat)

    return imgs_n2s, masks_n2s


def train_test_specialist(root, n_epochs=50, lr=5e-4, test=True):
    """
    Train a specialist model on noisy images and optionally test it.

        This method performs training on a dataset of noisy images and evaluates
        the model on a test set if specified. It involves loading training and
        validation data, training a neural network classifier, and performing
        cell segmentation on test images. The results are saved to a numpy file.

        Args:
            root: The root directory containing the dataset.
            n_epochs: The number of epochs for training the model.
            lr: The learning rate for the optimizer.
            test: A flag indicating whether to perform testing after training.

        Returns:
            If test is False, returns the validation loss as a float.
            If test is True, returns a tuple containing:
                - imgs: A list of processed images after segmentation.
                - masks_n2s: A list of segmentation masks corresponding to the images.
                - ap: Average precision scores calculated from the true and predicted masks.
    """
    n_train = 3 * 89 * 20
    n_val = 89 * 20

    dat = np.load(root / "noisy_test" / "test_poisson.npy", allow_pickle=True).item()
    test_noisy = dat["test_noisy"][:11]
    masks_true = dat["masks_true"][:11]
    diam_test = dat["diam_test"]

    im_train = [
        io.imread(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"))[
            np.newaxis, :, :
        ]
        for i in range(n_train)
    ]
    im_train.extend(test_noisy)
    im_val = [
        io.imread(Path(root / "noisy_test" / "care" / "source" / f"{i:03d}.tif"))
        for i in range(n_train, n_train + n_val)
    ]

    cells_train = Cells(im_train, xy=(128, 128))
    cells_val = Cells(np.array(im_val)[:, np.newaxis, :, :], xy=(128, 128))

    data_loader = DataLoader(cells_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(cells_val, batch_size=64, shuffle=True)

    loss_function = MSELoss()
    masker = Masker(width=4, mode="interpolate")

    model = Unet().to(device)
    tic = time.time()
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(data_loader):
            noisy_images = batch
            net_input, mask = masker.mask(noisy_images, i)
            net_output = model(net_input)
            loss = loss_function(net_output * mask, noisy_images * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += (loss.item()) * len(batch)
        train_loss /= len(cells_train) / len(cells_val)
        if ep < 10 or ep % 5 == 0 or ep == n_epochs - 1:
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    noisy_images = batch
                    net_input, mask = masker.mask(noisy_images, i)
                    net_output = model(net_input)
                    loss = loss_function(net_output * mask, noisy_images * mask)
                    val_loss += (loss.item()) * len(batch)
            # val_loss /= len(cells_val)
            print(
                f"Loss ( {ep} ): \t train: {train_loss:.3f}, val: {val_loss:.3f}, {time.time()-tic:.2f}s"
            )
    if not test:
        return val_loss
    else:

        imgs = []
        masks_n2s = []
        for i in range(len(test_noisy)):
            img = test_noisy[i].copy()
            img, ysub, xsub = transforms.pad_image_ND(img)
            img = torch.from_numpy(img).to(device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                simple_output = model(img)
            out = (
                simple_output.squeeze()[ysub[0] : ysub[-1] + 1, xsub[0] : xsub[-1] + 1]
                .cpu()
                .numpy()
            )

            imgs.append(out)
            seg_model = CellposeModel(gpu=True, model_type="cyto2")

            masks = seg_model.eval(
                out,
                diameter=diam_test[i],
                channels=[1, 0],
                channel_axis=0,
                normalize=True,
            )[0]

            masks_n2s.append(masks)

        for i in range(11):
            assert masks_true[i].shape == masks_n2s[i].shape

        ap, tp, fp, fn = metrics.average_precision(masks_true, masks_n2s)
        print(ap.mean(axis=0))

        dat[f"test_n2s"] = imgs
        dat[f"masks_n2s"] = masks_n2s

        np.save(root / "noisy_test" / f"test_poisson_n2s_specialist.npy", dat)

        return imgs, masks_n2s, ap
