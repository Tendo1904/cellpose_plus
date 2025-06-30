"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    """
    Creates a sequential neural network module that includes batch normalization, ReLU activation, and a convolutional layer.

    Args:
        in_channels: The number of input channels for the convolutional layer.
        out_channels: The number of output channels for the convolutional layer.
        sz: The size of the convolutional kernel.
        conv_3D: A boolean flag that determines whether to use 3D convolution (True) or 2D convolution (False).

    Returns:
        A sequential module containing batch normalization, ReLU activation, and the specified convolutional layer.
    """
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz, conv_3D=False):
    """
    Creates a sequential block of layers consisting of a batch normalization layer followed by a convolutional layer.

    Args:
        in_channels: The number of input channels for the convolutional layer.
        out_channels: The number of output channels for the convolutional layer.
        sz: The size of the convolutional kernel.
        conv_3D: A boolean flag indicating whether to use 3D convolution (True) or 2D convolution (False).

    Returns:
        A sequential container with the specified batch normalization and convolutional layer.
    """
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    """No valid docstring found."""

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        """
        Initializes the class with the specified parameters for a neural network model.

        This constructor sets up the base parameters, convolution options, and structure of the network,
        including downsampling and upsampling configurations.

        Args:
            nbase: A list of base channel sizes for the network.
            nout: The number of output channels for the network.
            sz: The size of the input to the network.
            mkldnn: A boolean indicating whether to use MKLDNN for performance optimization.
            conv_3D: A boolean indicating whether to use 3D convolutions.
            max_pool: A boolean indicating whether to apply max pooling.
            diam_mean: A float representing the mean diameter parameter for training.

        Returns:
            None
        """
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D)
        for t in range(4):
            if t == 0:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(in_channels, out_channels, sz, conv_3D)
                )
            else:
                self.conv.add_module(
                    "conv_%d" % t, batchconv(out_channels, out_channels, sz, conv_3D)
                )

    def forward(self, x):
        """
        Performs a forward pass through the upsampling layers using the provided style and input data.

        Args:
            style: The style information used for processing the input data.
            xd: A list of input tensors that are used in the upsampling process.
            mkldnn: A boolean flag indicating whether to use MKL-DNN for optimized performance.

        Returns:
            A tensor resulting from the upsampling and style processing.
        """
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    """
    A class that implements downsampling layers for neural networks, building a structure to reduce the spatial dimensions while preserving important features.

    Attributes:
    - nbase: A list of integers that defines the number of channels for each downsampling layer.
    - sz: A size parameter utilized in the construction of the residual modules for downsampling.
    - conv_3D: A boolean indicating whether 3D convolution layers are used.
    - max_pool: A boolean that determines if max pooling should be used instead of average pooling.

    Methods:
    - __init__
    - forward

    The attributes configure the behavior and architecture of the downsampling process. The methods facilitate initialization of the layers and the execution of the downsampling operations on input data.
    """

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        """
        Initializes an instance of the class and sets up the upsampling layers.

        This constructor initializes the upsampling process using a nearest neighbor approach. It creates a sequential model that consists of a series of residual upsampling layers based on the provided base sizes.

        Args:
            nbase: A list of base sizes used for determining the number of channels in the upsampling layers.
            sz: A size parameter used in the residual upsampling layers.
            conv_3D: A boolean flag indicating whether to use 3D convolution in the residual upsampling layers.

        Returns:
            None
        """
        super().__init__()
        self.down = nn.Sequential()
        if max_pool:
            self.maxpool = (
                nn.MaxPool3d(2, stride=2) if conv_3D else nn.MaxPool2d(2, stride=2)
            )
        else:
            self.maxpool = (
                nn.AvgPool3d(2, stride=2) if conv_3D else nn.AvgPool2d(2, stride=2)
            )
        for n in range(len(nbase) - 1):
            self.down.add_module(
                "res_down_%d" % n, resdown(nbase[n], nbase[n + 1], sz, conv_3D)
            )

    def forward(self, x):
        """

        Computes the forward pass of the model, generating a style representation from the input tensor.

        Parameters:
            x0: The input tensor from which the style representation is derived.

        Returns:
            A tensor representing the normalized style of the input.
        """
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class batchconvstyle(nn.Module):
    """No valid docstring found."""

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initialize the class by setting up the flattening layer and the appropriate average pooling function based on the conv_3D parameter.

        Args:
            conv_3D: A boolean that determines whether to use 3D average pooling or 2D average pooling.

        Returns:
            None
        """
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        """
        No valid docstring found.
        """
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y


class resup(nn.Module):
    """
    Resup is a Python class that implements a specific functionality related to handling data or processing tasks as defined by the application's requirements.

    Class Methods:
    - __init__:
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializer for the class, setting up the convolutional layers and projection layer.

        Args:
            in_channels: Number of input channels for the first convolutional layer.
            out_channels: Number of output channels for the convolutional layers.
            style_channels: Number of style channels used in the style convolutional layers.
            sz: Size parameter that defines the dimensions of the convolutional layers.
            conv_3D: A boolean flag indicating whether to use 3D convolutions.

        Returns:
            None
        """
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv_0", batchconv(in_channels, out_channels, sz, conv_3D=conv_3D)
        )
        self.conv.add_module(
            "conv_1",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.conv.add_module(
            "conv_2",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.conv.add_module(
            "conv_3",
            batchconvstyle(
                out_channels, out_channels, style_channels, sz, conv_3D=conv_3D
            ),
        )
        self.proj = batchconv0(in_channels, out_channels, 1, conv_3D=conv_3D)

    def forward(self, x, y, style, mkldnn=False):
        """
        Computes the forward pass for a model by combining input tensors and passing them through layers.

        Args:
            style: The style representation used to influence the output computation.
            x: The input tensor that will be processed.
            mkldnn: A boolean flag indicating whether to use MKL-DNN for optimized operations.
            y: An optional tensor that will be added to the input tensor `x`.

        Returns:
            The output tensor resulting from the forward pass through the model.
        """
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](
            style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn
        )
        return x


class make_style(nn.Module):
    """No valid docstring found."""

    def __init__(self, conv_3D=False):
        """
        Initializes an instance of the class.

        This constructor sets up the necessary layers for the model, including a batch convolution layer
        and a fully connected layer.

        Args:
            in_channels: The number of input channels for the batch convolution layer.
            out_channels: The number of output channels for both the batch convolution and the fully connected layer.
            style_channels: The number of input features for the fully connected layer.
            sz: The size parameter for the batch convolution layer.
            conv_3D: A boolean flag indicating whether to use 3D convolution (default is False).

        Returns:
            None
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        """
        Processes the input through a series of downsampling operations and returns the outputs.

        Args:
            x: The input data that will be processed through the downsampling layers.

        Returns:
            A list of outputs from each downsampling layer after applying max pooling where applicable.
        """
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


class upsample(nn.Module):
    """
    A class that implements an upsampling mechanism utilizing residual upsampling layers for enhanced image resolution.

    Attributes:
    - nbase: A list containing base sizes for determining the number of channels in the upsampling layers.
    - sz: A size parameter used in the residual upsampling layers.
    - conv_3D: A boolean flag that indicates the usage of 3D convolution in the upsampling layers.

    Methods:
    - __init__: Initializes an instance of the class and sets up the upsampling layers.
    - forward: Performs a forward pass through the upsampling layers using the provided style and input data.
    """

    def __init__(self, nbase, sz, conv_3D=False):
        """


        Initialization method for setting up the downsampling layers in the neural network architecture.



        Args:

            nbase: A list of integers representing the number of channels for each layer in the downsampling path.

            sz: A size parameter used in the construction of the residual downsampling modules.

            conv_3D: A boolean flag indicating whether to use 3D convolution layers (default is False).

            max_pool: A boolean flag indicating whether to use max pooling instead of average pooling (default is True).



        Returns:

            None

        """
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module(
                "res_up_%d" % (n - 1),
                resup(nbase[n], nbase[n - 1], nbase[-1], sz, conv_3D),
            )

    def forward(self, style, xd, mkldnn=False):
        """
        Computes the forward pass of the model, applying projections and convolution operations to the input.

        Args:
            x: The input data to be processed by the model.

        Returns:
            The transformed output after applying the projection and convolution layers.
        """
        x = self.up[-1](xd[-1], xd[-1], style, mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            x = self.up[n](x, xd[n], style, mkldnn=mkldnn)
        return x


class CPnet(nn.Module):
    """
    CPnet is the Cellpose neural network model used for cell segmentation and image restoration.

    Args:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        mkldnn (bool, optional): Whether to use MKL-DNN acceleration. Defaults to False.
        conv_3D (bool, optional): Whether to use 3D convolution. Defaults to False.
        max_pool (bool, optional): Whether to use max pooling. Defaults to True.
        diam_mean (float, optional): Mean diameter of the cells. Defaults to 30.0.

    Attributes:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        residual_on (bool): Whether to use residual connections.
        style_on (bool): Whether to use style transfer.
        concatenation (bool): Whether to use concatenation.
        conv_3D (bool): Whether to use 3D convolution.
        mkldnn (bool): Whether to use MKL-DNN acceleration.
        downsample (nn.Module): Downsample blocks of the network.
        upsample (nn.Module): Upsample blocks of the network.
        make_style (nn.Module): Style module, avgpool's over all spatial positions.
        output (nn.Module): Output module - batchconv layer.
        diam_mean (nn.Parameter): Parameter representing the mean diameter to which the cells are rescaled to during training.
        diam_labels (nn.Parameter): Parameter representing the mean diameter of the cells in the training set (before rescaling).

    """

    def __init__(
        self,
        nbase,
        nout,
        sz,
        mkldnn=False,
        conv_3D=False,
        max_pool=True,
        diam_mean=30.0,
    ):
        """
        Initializes the neural network module with convolutional layers.

        This constructor sets up the initial layers of the network, including a projection layer
        and a series of convolutional layers organized in a sequential manner. The number of layers
        and their configurations depend on the input and output channel counts, as well as the
        kernel size and whether 3D convolutions are used.

        Args:
            in_channels: Number of input channels for the first convolutional layer.
            out_channels: Number of output channels for the convolutional layers.
            sz: Size of the kernel for the convolutional layers.
            conv_3D: Boolean indicating whether to use 3D convolutions instead of 2D.

        Returns:
            None
        """
        super().__init__()
        self.nchan = nbase[0]
        self.nbase = nbase
        self.nout = nout
        self.sz = sz
        self.residual_on = True
        self.style_on = True
        self.concatenation = False
        self.conv_3D = conv_3D
        self.mkldnn = mkldnn if mkldnn is not None else False
        self.downsample = downsample(nbase, sz, conv_3D=conv_3D, max_pool=max_pool)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup, sz, conv_3D=conv_3D)
        self.make_style = make_style(conv_3D=conv_3D)
        self.output = batchconv(nbaseup[0], nout, 1, conv_3D=conv_3D)
        self.diam_mean = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )
        self.diam_labels = nn.Parameter(
            data=torch.ones(1) * diam_mean, requires_grad=False
        )

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def forward(self, data):
        """
        Forward pass of the CPnet model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        if self.mkldnn:
            data = data.to_mkldnn()
        T0 = self.downsample(data)
        if self.mkldnn:
            style = self.make_style(T0[-1].to_dense())
        else:
            style = self.make_style(T0[-1])
        style0 = style
        if not self.style_on:
            style = style * 0
        T1 = self.upsample(style, T0, self.mkldnn)
        T1 = self.output(T1)
        if self.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()
        return T1, style0, T0

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(
                self.nbase,
                self.nout,
                self.sz,
                self.mkldnn,
                self.conv_3D,
                self.diam_mean,
            )
            state_dict = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=True
            )

        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            self.load_state_dict(
                dict([(name, param) for name, param in state_dict.items()]),
                strict=False,
            )


class CPnetBioImageIO(CPnet):
    """
    A subclass of the CPnet model compatible with the BioImage.IO Spec.

    This subclass addresses the limitation of CPnet's incompatibility with the BioImage.IO Spec,
    allowing the CPnet model to use the weights uploaded to the BioImage.IO Model Zoo.
    """

    def forward(self, x):
        """
        Perform a forward pass of the CPnet model and return unpacked tensors.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        output_tensor, style_tensor, downsampled_tensors = super().forward(x)
        return output_tensor, style_tensor, *downsampled_tensors

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device, weights_only=True)
        else:
            self.__init__(
                self.nbase,
                self.nout,
                self.sz,
                self.mkldnn,
                self.conv_3D,
                self.diam_mean,
            )
            state_dict = torch.load(
                filename, map_location=torch.device("cpu"), weights_only=True
            )

        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary into the model.

        This method overrides the default `load_state_dict` to handle Cellpose's custom
        loading mechanism and ensures compatibility with BioImage.IO Core.

        Args:
            state_dict (Mapping[str, Any]): A state dictionary to load into the model
        """
        if state_dict["output.2.weight"].shape[0] != self.nout:
            for name in self.state_dict():
                if "output" not in name:
                    self.state_dict()[name].copy_(state_dict[name])
        else:
            super().load_state_dict(
                {name: param for name, param in state_dict.items()}, strict=False
            )
