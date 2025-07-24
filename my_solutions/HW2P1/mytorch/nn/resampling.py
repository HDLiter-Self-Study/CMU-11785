import numpy as np


class Upsample1d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        output_width = (A.shape[2] - 1) * self.upsampling_factor + 1
        # Get the empty Z
        Z = np.zeros((A.shape[0], A.shape[1], output_width))

        # Assign A to sliced Z
        Z[:, :, :: self.upsampling_factor] = A
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Slice dLdZ to get dLdA
        dLdA = dLdZ[:, :, :: self.upsampling_factor]

        return dLdA


class Downsample1d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        input_width = A.shape[-1]
        # We store the input width here because we need to know whether the origin size is odd or even when bp
        self.input_width = input_width

        # Slice A to get Z
        Z = A[:, :, :: self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        # Get the empty dLdA
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_width))

        # Assign dLdZ to sliced dLdA
        dLdA[:, :, :: self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, channels, input_height, input_width = A.shape
        output_height = (input_height - 1) * self.upsampling_factor + 1
        output_width = (input_width - 1) * self.upsampling_factor + 1

        # Get the empty Z
        Z = np.zeros((batch_size, channels, output_height, output_width))

        # Assign sliced A to Z
        Z[:, :, :: self.upsampling_factor, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Slice dLdZ to get dLdA
        dLdA = dLdZ[:, :, :: self.upsampling_factor, :: self.upsampling_factor]

        return dLdA


class Downsample2d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        # We store the input width and height here because we need to know whether the origin size is odd or even when bp
        self.input_height, self.input_width = A.shape[-2:]
        # Slice A to get Z
        Z = A[:, :, :: self.downsampling_factor, :: self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Get the empty dLdA
        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.input_height, self.input_width))

        # Assign dLdZ to sliced dLdA
        dLdA[:, :, :: self.downsampling_factor, :: self.downsampling_factor] = dLdZ

        return dLdA
