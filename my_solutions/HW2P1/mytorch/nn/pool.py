import numpy as np
from resampling import *


class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.A_windows = np.lib.stride_tricks.sliding_window_view(
            self.A, (self.kernel, self.kernel), (2, 3)
        )  # Shape (batch_size, in_channels, output_width, output_height, kernel, kernel)

        return np.max(self.A_windows, (4, 5))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Note that out_channels == in_channels, don't sum across channels

        # Find the max element in each window, and add the gradient from dLdZ to the max position

        # Flatten the windows to use argmax
        A_flattened_windows = self.A_windows.reshape(
            *self.A_windows.shape[:-2], -1
        )  # Shape (batch_size, in_channels, output_width, output_height, kernel * kernel)
        max_indices = np.argmax(A_flattened_windows, -1)  # Shape (batch_size, in_channels, output_width, output_height)

        # Convert flat indices to 2D coordinates
        max_col, max_row = np.unravel_index(max_indices, (self.kernel, self.kernel))

        # Create coordinate arrays for all positions
        batch_idx, channel_idx, out_col, out_row = np.ogrid[
            : dLdZ.shape[0], : dLdZ.shape[1], : dLdZ.shape[2], : dLdZ.shape[3]
        ]

        # Calculate actual positions in input array
        # Shape (batch_size, in_channels, output_width, output_height)
        input_row = out_row + max_row
        input_col = out_col + max_col

        # Initialize output and use advanced indexing to add gradients
        dLdA = np.zeros_like(self.A)
        np.add.at(dLdA, (batch_idx, channel_idx, input_col, input_row), dLdZ)

        return dLdA


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.A_windows = np.lib.stride_tricks.sliding_window_view(
            A, (self.kernel, self.kernel), (2, 3)
        )  # Shape (batch_size, in_channels, output_width, output_height, kernel, kernel)
        return np.mean(self.A_windows, (4, 5))

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # Forward process can be viewed as: A [convolute] mean_kernel -> Z
        mean_kernel = np.ones((self.kernel, self.kernel)) / (
            self.kernel * self.kernel
        )  #  Shape (kernel_size, kernel_size)

        # So padded_dLdZ [convolute] flipped_mean_kernel -> dLdA
        # P.S. flipped_mean_kernel == mean_kernel since kernel is symetric
        pad_size = (self.kernel - 1, self.kernel - 1)
        zero_pad = (0, 0)
        padded_dLdZ = np.pad(dLdZ, (zero_pad, zero_pad, pad_size, pad_size))

        # Do the convolution
        padded_dLdZ_windows = np.lib.stride_tricks.sliding_window_view(
            padded_dLdZ, mean_kernel.shape, (2, 3)
        )  # Shape (batch_size, out_channels, input_width, input_height, kernel_size, kernel_size)

        dLdA = np.tensordot(
            padded_dLdZ_windows, mean_kernel, ((4, 5), (0, 1))
        )  # Shape (batch_size, out_channels, input_width, input_height)

        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        unsampled_Z = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(unsampled_Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        reverse_sampled_dLdZ = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(reverse_sampled_dLdZ)

        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MeanPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        unsampled_Z = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(unsampled_Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        reverse_sampled_dLdZ = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(reverse_sampled_dLdZ)

        return dLdA
