"""
CNN Network model for HW2P2: Image Recognition and Verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionBlock(torch.nn.Module):
    """Basic convolution block with Conv2d + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Network(torch.nn.Module):
    """CNN Network for face recognition and verification"""

    def __init__(self, num_classes=8631):
        super().__init__()

        self.backbone = torch.nn.Sequential(
            ConvolutionBlock(3, 64, 7, 4, 3),
            ConvolutionBlock(64, 128, 3, 2, 1),
            ConvolutionBlock(128, 256, 3, 2, 1),
            ConvolutionBlock(256, 512, 3, 2, 1),
            ConvolutionBlock(512, 1024, 3, 2, 1),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = torch.nn.Flatten()
        self.cls_layer = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        feats = []
        for layer in self.backbone.children():
            x = layer(x)
            feats.append(x)

        flattened_x = self.flatten(x)
        out = self.cls_layer(flattened_x)

        return {"feats": flattened_x, "all_feats": feats, "out": out}
