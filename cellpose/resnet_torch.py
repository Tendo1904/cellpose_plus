"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def batchconv(in_channels, out_channels, sz, conv_3D=False):
    """
    Creates a sequential model consisting of batch normalization, ReLU activation, and a convolution layer.

        This method constructs a neural network layer sequence that includes batch normalization,
        an activation function, and either 2D or 3D convolution. The specific layer types are
        determined by the `conv_3D` parameter.

        Args:
            in_channels: Number of input channels for the convolution layer.
            out_channels: Number of output channels for the convolution layer.
            sz: Size of the convolution kernel.
            conv_3D: A boolean indicating whether to use 3D convolution layers (True) or 2D convolution layers (False).

        Returns:
            A Sequential model containing the batch normalization layer, ReLU activation,
            and the convolution layer configured according to the input parameters.
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
    Creates a sequential model consisting of a batch normalization layer followed by a convolution layer.

    This method constructs a neural network module that includes either a 2D or 3D convolution layer
    based on the specified parameters. The convolution layer is preceded by a batch normalization layer
    which helps in stabilizing the learning process.

    Args:
        in_channels: The number of input channels for the convolution layer.
        out_channels: The number of output channels (filters) for the convolution layer.
        sz: The size of the convolution kernel.
        conv_3D: A boolean flag indicating whether to use 3D convolution (default is False for 2D).

    Returns:
        A sequential model containing the batch normalization layer followed by the convolution layer.
    """
    conv_layer = nn.Conv3d if conv_3D else nn.Conv2d
    batch_norm = nn.BatchNorm3d if conv_3D else nn.BatchNorm2d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


class resdown(nn.Module):
    """
    A class that implements a convolutional neural network layer for downsampling input data.

    This class is designed to initialize and manage a series of convolutional operations that
    project input channels to output channels while optionally supporting 3D convolutions for
    more complex data representations.

    Attributes:
        in_channels: The number of input channels for the first layer.
        out_channels: The number of output channels for the layers.
        sz: The size of the convolutional kernel.
        conv_3D: A boolean flag indicating whether to utilize 3D convolutions.

    Methods:
        __init__: Initializes the convolutional layers with specified parameters.
        forward: Computes the forward pass of the model by processing the input tensor.
    """

    def __init__(self, in_channels, out_channels, sz, conv_3D=False):
        """
        Initializes the convolutional neural network layer.

            This method sets up a sequence of convolutional layers for the neural
            network, projecting the input channels to output channels and
            applying a series of convolutional operations, with an option for 3D
            convolutions.

            Args:
                in_channels: The number of input channels for the first layer.
                out_channels: The number of output channels for the layers.
                sz: The size of the convolutional kernel.
                conv_3D: A boolean flag indicating whether to use 3D convolutions
                          instead of the default 2D convolutions.

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
        Computes the forward pass of the model.

            This method takes an input tensor, applies a series of transformations
            using projection and convolutional layers, and returns the transformed tensor.

            Args:
                x: The input tensor to be processed.

            Returns:
                The transformed tensor after applying the projection and convolutional layers.
        """
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    """
    Downsampling module with residual layers and pooling options.

        This class implements a downsampling module that includes a series of
        residual downsampling layers. The downsampling process can be
        customized to use either max pooling or average pooling based on the
        parameters provided during initialization.

        Attributes:
            nbase: A list representing the number of channels for each layer
                   in the downsampling path.
            sz: A size parameter for the residual blocks, indicating the input size.
            conv_3D: A flag that determines whether to use 3D convolution layers.
            max_pool: A flag that indicates the pooling method used (max or average).

        Methods:
            __init__: Initializes the downsampling module with specified parameters.
            forward: Performs a forward pass through the network, processing
                     an input tensor through the defined layers.
    """

    def __init__(self, nbase, sz, conv_3D=False, max_pool=True):
        """
        Initializes the downsampling module.

            This method sets up a downsampling module consisting of a series of
            residual downsampling layers, optionally followed by either max pooling
            or average pooling depending on the parameters provided.

            Args:
                nbase: A list of integers representing the number of channels
                        for each layer in the downsampling path.
                sz: A size parameter for the residual blocks, typically indicating
                    the input size.
                conv_3D: A boolean flag indicating whether to use 3D convolution
                          layers (True) or 2D convolution layers (False).
                max_pool: A boolean flag indicating whether to use max pooling (True)
                           or average pooling (False) for downsampling.

            Returns:
                None: This method does not return a value. It initializes the
                      internal state of the object.
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
        Perform a forward pass through the network.

            This method takes an input tensor and passes it through a series of
            layers defined in the 'down' attribute of the class, applying max pooling
            as necessary between layers. The output is a list of tensors, with each
            tensor corresponding to the output of each layer.

            Args:
                x: The input tensor to the network.

            Returns:
                A list of tensors, where each tensor is the output from the
                corresponding layer in the 'down' attribute after processing
                the input through optional max pooling.
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
    A model that performs batch convolution with optional style transformations.

    This class implements a convolutional neural network that can process input data with an optional
    style parameter, allowing for dynamic modifications of the input during the forward pass.

    Methods:
        __init__: Initializes the model with specified parameters.
        forward: Computes the forward pass of the model by processing input data with optional style.

    Attributes:
        in_channels: The number of input channels for the convolutional layer.
        out_channels: The number of output channels for the convolutional layer.
        style_channels: The number of input features for the linear layer.
        sz: The size parameter used for the convolutional layer.
        conv_3D: A boolean indicating whether to use 3D convolution.

    The __init__ method sets up the necessary layers for the model based on the provided parameters.
    The forward method takes input tensors and applies a style transformation with optional MKLDNN
    optimization for improved performance.
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes the model with specified parameters.

            This method sets up the necessary layers for the model, including
            convolutional and linear layers based on the provided input parameters.

            Args:
                in_channels: The number of input channels for the convolutional layer.
                out_channels: The number of output channels for the convolutional layer.
                style_channels: The number of input features for the linear layer.
                sz: The size parameter used for the convolutional layer.
                conv_3D: A boolean indicating whether to use 3D convolution (default is False).

            Returns:
                None
        """
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz, conv_3D)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        """
        Computes the forward pass of the model by processing input data with optional style.

            This method takes an input tensor and an optional additional tensor, applies a style transformation,
            and performs convolution on the resulting tensor. The method also supports a MKLDNN optimized path
            for performance improvements when specified.

            Args:
                style: The style parameter used for the transformation.
                x: The primary input tensor to be processed.
                mkldnn: Indicates whether to use MKLDNN for optimized computation (default is False).
                y: An optional additional tensor to be added to the primary input before processing.

            Returns:
                A tensor that represents the output of the convolution operation after
                processing the input and applying the style transformation.
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
    A convolutional neural network module designed for processing input data
    through a series of convolutional layers, incorporating style channels for
    enhanced feature representation.

    Methods:
        __init__: Initializes the convolutional neural network architecture.
        forward: Computes the forward pass to process input data and
                 auxiliary tensors.

    Attributes:
        in_channels: The number of input channels for the first convolution layer.
        out_channels: The number of output channels for the convolution layers.
        style_channels: The number of style channels utilized in the convolution layers.
        sz: The spatial size of the convolutions.
        conv_3D: A flag indicating the use of 3D convolutions.

    The __init__ method sets up the neural network layers according to the
    specified input and output channels, alongside any style information
    necessary for processing. The forward method handles the transformation
    of input data, applying convolutions and projections based on the
    provided auxiliary and style information, and returns the processed output.
    """

    def __init__(self, in_channels, out_channels, style_channels, sz, conv_3D=False):
        """
        Initializes a convolutional neural network module.

            This method sets up the neural network architecture by defining the
            convolutional layers involved in the processing of input data. It
            includes a series of batch convolutions that handle the transformation
            of input channels, while considering additional styling channels if
            specified.

            Args:
                in_channels: The number of input channels for the first convolution layer.
                out_channels: The number of output channels for the convolution layers.
                style_channels: The number of style channels used in the convolution layers.
                sz: The spatial size of the convolutions.
                conv_3D: A boolean indicating whether to use 3D convolutions (default is False).

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
        Computes the forward pass of the model.

            This method applies a series of convolutional operations to the input
            data along with additional projections and computations based on the
            style and auxiliary input. It returns the processed output after the
            transformations.

            Args:
                x: The input tensor to be processed.
                y: An auxiliary tensor used in the computation.
                style: A tensor representing the style information to guide the
                    transformations.
                mkldnn: A boolean flag indicating whether to use MKL-DNN optimization
                    for convolution operations.

            Returns:
                The resulting tensor after applying the series of convolutions and
                transformations.
        """
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](
            style, self.conv[2](style, x, mkldnn=mkldnn), mkldnn=mkldnn
        )
        return x


class make_style(nn.Module):
    """
    A class used to create a normalized style representation from input tensors.

    This class implements methods for initializing and processing input tensors
    to generate a style representation through average pooling and normalization.

    Methods:
        __init__
        forward

    Attributes:
        conv_3D

    The __init__ method initializes the object with a specified pooling option
    (either 2D or 3D). The forward method processes an input tensor, reducing its dimensions
    via average pooling, flattening the output, and normalizing it to achieve the final style representation.
    """

    def __init__(self, conv_3D=False):
        """
        Initializes an instance of the class.

            This method sets up the necessary components of the object,
            including a flatten layer and a pooling function based on the
            specified parameter.

            Args:
                conv_3D: A boolean flag indicating whether to use 3D
                    average pooling. If True, 3D pooling will be used;
                    otherwise, 2D pooling will be utilized.

            Returns:
                None
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d if conv_3D else F.avg_pool2d

    def forward(self, x0):
        """
        Processes the input tensor to produce a normalized style representation.

            This method takes an input tensor, applies average pooling to reduce its dimensions,
            flattens the result, and normalizes it.

            Args:
                x0: The input tensor to be processed.

            Returns:
                A tensor representing the normalized style derived from the input.
        """
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


class upsample(nn.Module):
    """
    A class for performing upsampling operations using a series of residual blocks.

    This class sets up an upsampling module and executes a forward pass that applies
    transformations and upsampling to input data, enhancing feature representation
    through skip connections.

    Methods:
        __init__: Initializes the upsampling module.
        forward: Computes the forward pass of the model.

    Attributes:
        None

    The __init__ method sets up the necessary components for the upsampling,
    including a sequential model consisting of multiple residual upsampling blocks,
    based on the provided base network structure. The forward method performs the
    actual computation by applying a sequence of upsampling and transformation operations
    to the input data, utilizing style information and skip connections.
    """

    def __init__(self, nbase, sz, conv_3D=False):
        """
        Initializes the upsampling module.

            This method sets up an upsampling layer and a sequential model comprising
            multiple residual upsampling blocks based on the provided base network structure.

            Args:
                nbase: A list of integers representing the number of channels for each level
                    in the base network.
                sz: An integer representing the size of the input tensors.
                conv_3D: A boolean flag indicating whether to use 3D convolutions
                    (default is False).

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

            This method applies a series of upsampling and transformation operations
            to the input data, modified by a specified style. The computations
            are performed in a loop that iterates over a list of layers,
            combining the outputs with corresponding inputs from a skip connection
            to enhance feature representation.

            Args:
                style: The style information to be applied during transformations.
                xd: A list containing the inputs for skip connections at each layer.
                mkldnn: A boolean flag indicating whether to use MKLDNN for optimized performance.

            Returns:
                The result of the forward pass, which is the output after all transformations
                and upsampling operations have been applied.
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
        Initializes a neural network module with customizable parameters.

            This constructor sets up the initial configuration for the neural network,
            including the number of channels, layers, and various options for
            convolution and pooling operations. It defines parameters for upsampling
            and style, and also initializes specific layers according to the
            provided parameters.

            Args:
                nbase: A list or array representing the number of base channels in
                        each layer of the network.
                nout: The number of output channels for the final layer.
                sz: The size of the input to the network.
                mkldnn: A boolean indicating whether to use MKL-DNN for optimized performance
                         (default is False).
                conv_3D: A boolean indicating whether to use 3D convolution (default is False).
                max_pool: A boolean indicating whether to use max pooling in the downsampling
                           process (default is True).
                diam_mean: A float representing the mean diameter value for certain operations
                           in the network (default is 30.0).

            Returns:
                None. This method initializes the instance of the class but does not return
                any value.
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
