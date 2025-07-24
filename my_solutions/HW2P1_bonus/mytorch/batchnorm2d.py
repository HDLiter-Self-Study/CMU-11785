# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        if eval:
            # eval mode - normalize with running stats, but don't overwrite training NZ or BZ
            # Otherwise there will be error in runner.py
            NZ_eval = (self.Z - self.running_M) / np.sqrt(self.eps + self.running_V)
            BZ_eval = NZ_eval * self.BW + self.Bb
            return BZ_eval

        self.Z = Z
        self.N = self.Z.shape[0] * self.Z.shape[2] * self.Z.shape[3]

        # training mode - compute batch stats and normalize with them
        self.M = np.mean(self.Z, (0, 2, 3), keepdims=True)
        self.V = np.var(self.Z, (0, 2, 3), keepdims=True)
        self.NZ = (self.Z - self.M) / np.sqrt(self.eps + self.V)
        self.BZ = self.NZ * self.BW + self.Bb
        # Update running statistics
        self.running_M = self.running_M * self.alpha + self.M * (1 - self.alpha)
        self.running_V = self.running_V * self.alpha + self.V * (1 - self.alpha)

        return self.BZ

    def backward(self, dLdBZ):

        dBZdBW = self.NZ
        dBZdBb = np.ones_like(self.Bb)
        self.dLdBW = np.sum(dLdBZ * dBZdBW, axis=(0, 2, 3), keepdims=True)
        self.dLdBb = np.sum(dLdBZ * dBZdBb, axis=(0, 2, 3), keepdims=True)

        dBZdNZ = self.BW
        dLdNZ = dLdBZ * dBZdNZ

        dNZdV = (self.Z - self.M) * (-0.5) * np.pow((self.eps + self.V), -1.5)
        dLdV = np.sum(dLdNZ * dNZdV, axis=(0, 2, 3), keepdims=True)

        dNZdM = -1 / np.sqrt(self.eps + self.V)
        dLdM = np.sum(dLdNZ * dNZdM, axis=(0, 2, 3), keepdims=True)

        dNZdZ = 1 / np.sqrt(self.eps + self.V)
        dMdZ = np.ones_like(self.Z) / self.N
        dVdZ = (self.Z - self.M) * 2 / self.N
        dLdZ = dLdV * dVdZ + dLdM * dMdZ + dLdNZ * dNZdZ

        return dLdZ
