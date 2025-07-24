import numpy as np
from resampling import *


class Conv2d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def _2d_stride1_convolute(self, input, filters):
        """
        Argument:
            input (np.array): (batch_size, in_channels, input_height, input_width)
            filters (np.array): (out_channels, in_channels, kernel_size, kernel_size)
        Return:
            output (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Get kernel_size
        _, _, kernel_size, _ = filters.shape

        # Get windows
        input_windows = np.lib.stride_tricks.sliding_window_view(
            input, window_shape=(kernel_size, kernel_size), axis=(2, 3)
        )  # Shape (batch_size, in_channels, output_height, output_width, kernel_size, kernel_size)

        # Do the convolute
        output = np.tensordot(
            input_windows, filters, axes=((1, 4, 5), (1, 2, 3))
        )  # Result shape (batch_size, output_height, output_width, out_channels)
        output = np.transpose(
            output, axes=(0, 3, 1, 2)
        )  # Shape (batch_size, out_channels, output_height, output_width)

        return output

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        # Convolute A with W to get Z
        Z = self._2d_stride1_convolute(A, self.W)

        # Add the bias
        Z += self.b.reshape(1, -1, 1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Sum the filter windows and batches to get dLdb
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # Shape (out_channels,)

        # Convolute A with dLdZ to get dLdW
        # Treat "batch_size" as in_channels and "in_channels" as batch_size to sum over batches
        # Achieve this by flipping
        self.dLdW = self._2d_stride1_convolute(
            np.transpose(self.A, axes=(1, 0, 2, 3)),  # Shape (in_channels, batch_size, input_height, input_width)
            np.transpose(dLdZ, axes=(1, 0, 2, 3)),  # Shape (out_channels, batch_size, output_height, output_width)
        )  # Result shape (in_channels, out_channels, kernel_size, kernel_size)
        self.dLdW = np.transpose(
            self.dLdW, axes=(1, 0, 2, 3)
        )  # Shape (out_channels, in_channels, kernel_size, kernel_size)

        # Get padded dLdZ and flipped W
        pad_size = (self.kernel_size - 1, self.kernel_size - 1)
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), pad_size, pad_size))
        flipped_W = self.W[:, :, ::-1, ::-1]  # Shape (out_channels, in_channels, kernel_size, kernel_size)
        # Convolute padded dLdZ with flipped W to get dLdA
        # Swith in_channel and out_channel in flipped_W for backward convolution
        dLdA = self._2d_stride1_convolute(padded_dLdZ, np.transpose(flipped_W, axes=(1, 0, 2, 3)))

        return dLdA


class Conv2d:
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None
    ):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        pad_size = (self.pad, self.pad)
        padded_A = np.pad(A, ((0, 0), (0, 0), pad_size, pad_size))

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(padded_A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        unsampled_dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv2d_stride1 backward
        padded_dLdA = self.conv2d_stride1.backward(unsampled_dLdZ)

        # Unpad the gradient
        if self.pad == 0:
            dLdA = padded_dLdA
        else:
            dLdA = padded_dLdA[:, :, self.pad : -self.pad, self.pad : -self.pad]

        return dLdA
