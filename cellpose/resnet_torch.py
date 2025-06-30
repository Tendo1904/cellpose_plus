"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    """
    No valid docstring found.

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
    No valid docstring found.

    """

    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    """
    A class for implementing a residual down-sampling block for neural networks.

    Attributes:
    - in_channels: The number of input channels for the convolutional layers.
    - out_channels: The number of output channels for the convolutional layers.
    - kernel_size: The size of the convolutional kernel.
    - use_3d: A flag indicating whether 3D convolution is used.

    Methods:
    - __init__
    - forward

    The __init__ method initializes the residual down-sampling block with parameters that define the input channels, output channels, kernel size, and whether 3D convolutions are to be used. The forward method processes the input data through the defined convolutions and projections, returning the transformed output.
    """

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        """
        Initialize the upsampling module with a sequence of residual upsamples.

        This method sets up an upsampling layer to increase the spatial resolution of input features.
        It constructs a sequential module of residual upsampling blocks based on the provided base
        channel sizes and output size.

        Args:
            nbase: A list of base channel sizes used to define the number of features in each residual
                upsample block.
            sz: The size of the output feature map after upsampling.
            conv_3D: A boolean flag indicating whether to use 3D convolution (default is False).

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
        Performs a forward pass through the network, applying upsampling and transformations based on the provided style.

        Args:
            style: The style information to be used during the transformation.
            xd: A list of input tensors for each layer.
            mkldnn: A boolean flag indicating whether to use MKL-DNN for performance optimization.

        Returns:
            A tensor resulting from the forward pass through the network.
        """
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    """No valid docstring found."""

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        """
        Initializes a new instance of the class.

        This constructor sets up the necessary components for the class, including
        a flatten layer and an average pooling function, which can operate in either
        2D or 3D mode based on the provided flag.

        Args:
            conv_3D: A boolean flag indicating whether to use 3D average pooling
                      (True) or 2D average pooling (False).

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
        Calculates the normalized style representation of the input tensor.

        This method takes an input tensor, applies average pooling to extract style features,
        flattens the pooled output, and normalizes it to create a style representation.

        Args:
            x0: The input tensor from which to derive the style representation.

        Returns:
            A tensor representing the normalized style features.
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
    """
    BatchConvStyle processes input tensors using convolutional layers while applying style transfer.

    Attributes:
    - Some attributes used for managing style transfer and convolutional settings.

    Methods:
    - __init__:
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes the neural network module with a series of convolutional layers.

        This constructor sets up the convolutional layers needed for the processing of input
        data during the forward pass. It also initializes a projection layer for transforming
        the input channels into the output channels.

        Args:
            in_channels: The number of input channels for the first convolution layer.
            out_channels: The number of output channels for the convolution layers.
            style_channels: The number of style channels used in the style convolution layers.
            sz: The size parameter used for the convolutional layers.
            conv_3D: A boolean flag indicating whether to use 3D convolutions (default is False).

        Returns:
            None
        """
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        """
        Processes the input tensors through a series of convolutional and projection layers.

        Args:
            x: The input tensor to be processed.
            y: An additional tensor used in the convolution operation.
            style: A tensor representing the style information for the convolution operations.
            mkldnn: A boolean flag to indicate whether to use MKLDNN optimization.

        Returns:
            The processed tensor resulting from the forward pass through the network.
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
    Resup is a class that facilitates the implementation of a reinforcement learning model, specializing in support for various environment interactions.

    Class Methods:
    - __init__:
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes the neural network layer with the specified parameters.

        This method sets up the convolutional layer and the fully connected layer based on the input and output channels, as well as the style channels.

        Args:
            in_channels: The number of input channels for the convolutional layer.
            out_channels: The number of output channels for the convolutional layer.
            style_channels: The number of channels of the style input for the fully connected layer.
            sz: The size of the convolution kernel.
            conv_3D: A boolean indicating whether to use 3D convolution (default is False).

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
        Computes the forward pass for the model, potentially incorporating a style feature.

        This method combines an input tensor `x` with additional features generated from
        the specified `style`. If provided, it adds another tensor `y` to `x` before processing.
        The resulting tensor is then passed through a convolutional layer.

        Args:
            style: The style representation used for feature extraction.
            x: The input tensor to which the style features are added.
            mkldnn: A boolean indicating whether to convert the tensor to MKLDNN format.
            y: An optional tensor that, if provided, will be added to the input tensor `x`.

        Returns:
            A tensor that is the result of adding the features to `x` and processing through
            the convolutional layer.
        """
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](
            style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn
        )
        return x


class make_style(nn.Module):
    """
    A class that provides functionality to create and manage styles for graphical elements.

    Attributes:
    - styles: A collection of style settings applied to graphical elements.
    - default_style: The default style setting used when no specific style is provided.

    The class enables the customization of appearance attributes for user interface components, allowing developers to define styles in a structured manner.
    """

    def __init__(self, conv_3D=False):
        """
        Initializes the downsampling block of a neural network.

        This method sets up a sequential model for downsampling, which can utilize either max pooling or average pooling based on the provided parameters. The downsampling modules are created based on the given base dimensions.

        Args:
            nbase: A list or array of base dimensions for the downsampling layers.
            sz: The size of the input to determine the output dimensions.
            conv_3D: A boolean indicating whether to use 3D convolutions (default is False).
            max_pool: A boolean indicating whether to use max pooling (default is True).

        Returns:
            None
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        """
        Processes the input through a series of operations defined in the 'down' layers.

        This method takes an input tensor and sequentially applies a series of transformations
        defined in the 'down' attribute, storing the results of each transformation in a list.
        If the current transformation is not the first, max pooling is applied to the output
        of the previous transformation before the current transformation is executed.

        Args:
            x: The input tensor to be processed.

        Returns:
            A list of tensors, where each tensor is the output of a transformation applied
            by the corresponding layer in 'down'.
        """
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


class upsample(nn.Module):
    """
    Upsample is a class that implements an upsampling module for neural networks, designed to increase the spatial dimensions of the input tensors while maintaining important feature information through residual connections.

    Attributes:
    - nbase: A list indicating the number of channels for each layer in the upsampling module.
    - sz: A size parameter that influences the configuration of the residual upsampling modules.
    - conv_3D: A boolean that specifies the use of 3D convolution in the upsampling process.

    Methods:
    - __init__: Initializes the upsampling module for a neural network.
    - forward: Computes the forward pass of the model using the specified style and input tensors.
    """

    def __init__(self, nbase, sz, conv_3D=False):
        """
        Initializes the neural network module with a series of convolutional layers.

        This constructor sets up the basic structure of the module, including projection and
        convolution layers based on the specified input and output channels, size, and dimensionality
        of the convolution.

        Args:
            in_channels: The number of input channels for the first convolution layer.
            out_channels: The number of output channels for the convolution layers.
            sz: The size of the convolutional kernel.
            conv_3D: A boolean flag indicating whether to use 3D convolutions.

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
        Computes the forward pass of the model.

        This method takes an input tensor, applies a series of projections and convolutions to transform the input, and returns the result.

        Args:
            x: The input tensor to be processed through the model.

        Returns:
            The transformed tensor after applying the forward pass operations.
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
        Initializes the CPnet class by configuring essential parameters such as the base number of channels, output channels, and input size. This setup defines the network architecture and specifies options for downsampling, upsampling, and style feature handling, preparing the model for optimized data processing.

        This constructor sets up the base channel, output number, size, and various flags for
        downsampling, upsampling, and styling features. It also creates parameters for diameter
        mean values.

        Args:
            nbase: A list containing the base number of channels for the network.
            nout: The number of output channels for the final layer of the network.
            sz: The size of the input to the network.
            mkldnn: A boolean flag indicating whether to use MKL-DNN optimizations.
            conv_3D: A boolean flag indicating if 3D convolutions should be used.
            max_pool: A boolean flag indicating whether max pooling should be applied.
            diam_mean: A float value representing the mean diameter initialized for parameters.

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
        Obtain the current device on which the model's parameters are allocated.

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
        Save the model's current parameters to a designated file, enabling later retrieval and utilization.

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
        Performs a forward pass through the network, returning the resulting output along with corresponding style information and downsampled representations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.


        """

        output_tensor, style_tensor, downsampled_tensors = super().forward(x)
        return output_tensor, style_tensor, *downsampled_tensors

    def load_model(self, filename, device=None):
        """
        Retrieve and initialize a model from a file, accommodating specific hardware configurations if provided.

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
        Load the specified state dictionary into the model while accommodating special requirements for parameter compatibility. This method customizes the loading process to ensure proper integration of parameters and enhances the model's functionality.

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
