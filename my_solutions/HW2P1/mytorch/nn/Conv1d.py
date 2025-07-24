# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1:
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    @staticmethod
    def _1d_stride1_convolute(input, filters):
        """
        Argument:
            input (np.array): (batch_size, in_channels, input_size)
            filters (np.array): (out_channels, in_channels, kernel_size)
        Return:
            output (np.array): (batch_size, out_channels, output_size)
        """
        # Get kernel_size
        _, _, kernel_size = filters.shape
        # Get windows
        input_windows = np.lib.stride_tricks.sliding_window_view(input, (kernel_size), (2))

        output = np.tensordot(
            input_windows,  # Shape (batch_size, in_channels, output_size, kernel_size)
            filters,  # Shape (out_channels, in_channels, kernel_size)
            axes=((1, 3), (1, 2)),
        )  # Result shape  (batch_size, output_size, out_channels)

        output = np.transpose(output, (0, 2, 1))  # Shape  (batch_size, out_channels, output_size)
        return output

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        # Do the convolution
        Z = self._1d_stride1_convolute(A, self.W)

        # Add the bias
        Z += self.b.reshape(1, -1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # Pad kernel_size - 1 around the output_size dim of dLdZ
        # Shape (batch_size, out_channels, output_size + self.kernel_size * 2 - 2)
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)))

        # Get flipped W
        flipped_W = self.W[:, :, ::-1]  # Shape (out_channels, in_channels, kernel_size)
        flipped_W = np.transpose(flipped_W, axes=(1, 0, 2))  # Shape (in_channels, out_channels, kernel_size)
        # Flipped the order of in and out channel since we a doing a backprop

        # Convolute padded_dLdZ by flipped W to get dLdA
        dLdA = self._1d_stride1_convolute(padded_dLdZ, flipped_W)

        # dZdb are the axis sum of different batchs of kernels
        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # Shape (out_channels,)

        # Convolute A by dLdZ to get dLdW (out_channels, in_channels, kernel_size)
        # Treat the batch_size as "in_channels" and in_channels as "batch_size"
        # By transposing the input and kernel
        self.dLdW = self._1d_stride1_convolute(
            np.transpose(self.A, (1, 0, 2)),  # Shape (in_channels, batch_size, input_size)
            np.transpose(dLdZ, (1, 0, 2)),  # Shape (out_channels, batch_size, output_size)
        )  # Result shape (in_channels, out_channels, kernel_size)
        self.dLdW = np.transpose(self.dLdW, axes=(1, 0, 2))  # Shape (out_channels, in_channels, kernel_size)

        return dLdA


class Conv1d:
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            weight_init_fn=weight_init_fn,
            bias_init_fn=bias_init_fn,
        )
        self.downsample1d = Downsample1d(downsampling_factor=stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        padded_A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        stride1_Z = self.conv1d_stride1.forward(padded_A)

        # downsample
        Z = self.downsample1d.forward(stride1_Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdY = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        padded_dLdA = self.conv1d_stride1.backward(dLdY)

        # Unpad the gradient
        if self.pad == 0:
            dLdA = padded_dLdA
        else:
            dLdA = padded_dLdA[:, :, self.pad : -self.pad]

        return dLdA
