import sys

sys.path.append("mytorch")

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layers = []

        conv2d_layer = Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.layers.append(conv2d_layer)

        batchnorm2d_layer = BatchNorm2d(out_channels)
        self.layers.append(batchnorm2d_layer)

    def forward(self, A):
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class ResBlock(object):

    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):

        self.convolution_layers = []
        self.convolution_layers.append(ConvBlock(in_channels, out_channels, filter_size, stride, padding))
        self.convolution_layers.append(ReLU())
        self.convolution_layers.append(ConvBlock(out_channels, out_channels, 1, 1, 0))

        self.final_activation = ReLU()

        if stride != 1 or in_channels != out_channels or filter_size != 1 or padding != 0:
            # residual branch is a “ConvBlock”
            self.residual_connection = ConvBlock(in_channels, out_channels, filter_size, stride, padding)
        else:
            # By default, the residual branch will be an “Identity function”.
            self.residual_connection = Identity()

    def forward(self, A):
        A_conv = A
        for layer in self.convolution_layers:
            A_conv = layer.forward(A_conv)

        A_resi = self.residual_connection.forward(A)

        A_final = self.final_activation.forward(A_resi + A_conv)

        return A_final

    def backward(self, grad):
        grad_A_final = self.final_activation.backward(grad)

        grad_A_resi = self.residual_connection.backward(grad_A_final)

        grad_A_conv = grad_A_final
        for layer in reversed(self.convolution_layers):
            grad_A_conv = layer.backward(grad_A_conv)

        return grad_A_resi + grad_A_conv
