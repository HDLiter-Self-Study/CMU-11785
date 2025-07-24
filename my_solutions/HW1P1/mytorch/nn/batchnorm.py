import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z  # Shape (N, C)
        self.N = Z.shape[0]
        self.M = np.mean(Z, axis=0)  # Shape (1, C)
        self.V = np.var(Z, axis=0)  # Shape (1, C)

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / np.sqrt(self.eps + self.V)  # Shape (N, C)
            self.BZ = self.NZ * self.BW + self.Bb  # Shape (N, C)

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            self.NZ = (Z - self.running_M) / np.sqrt(self.eps + self.running_V)  # Shape (N, C)
            self.BZ = self.NZ * self.BW + self.Bb  # Shape (N, C)

        return self.BZ

    def backward(self, dLdBZ):

        # LdBZ Shape (N, C)
        dBZdBW = self.NZ  # Shape (N,C), N derivatives in rows
        dBbdBZ = np.ones_like(self.Bb)
        self.dLdBW = np.sum(dLdBZ * dBZdBW, axis=0)  # Shape (1,C), add the N derivatives together
        self.dLdBb = np.sum(dLdBZ * dBbdBZ, axis=0)

        dBZdNZ = self.BW
        dLdNZ = dLdBZ * dBZdNZ  # Shape (N, C)

        dNZdV = (self.Z - self.M) * (-0.5) * np.pow((self.eps + self.V), -1.5)  # Shape (N, C)
        dLdV = np.sum(dLdNZ * dNZdV, axis=0)  # Shape (1,C), add the N derivatives together

        dNZdM = -1 / np.sqrt(self.eps + self.V)  # Shape (N, 1)
        dLdM = np.sum(dLdNZ * dNZdM, axis=0)  # Shape (1,C), add the N derivatives together

        dNZdZ = 1 / np.sqrt(self.eps + self.V)  # Shape (N, 1)
        dMdZ = 1 / self.N
        dVdZ = 2 * (self.Z - self.M) / self.N  # Shape (N, C)
        dLdZ = dLdNZ * dNZdZ + dLdV * dVdZ + dLdM * dMdZ

        return dLdZ
