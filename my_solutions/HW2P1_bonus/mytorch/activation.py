import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdZ):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dLdZ * dAdZ


class Sigmoid:

    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self, dLdZ):

        dAdZ = self.A - self.A * self.A

        return dLdZ * dAdZ


class Tanh:

    def forward(self, Z):

        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdZ):

        dAdZ = 1 - self.A * self.A

        return dLdZ * dAdZ


class ReLU:

    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdZ):

        dAdZ = np.where(self.A > 0, 1, 0)

        return dLdZ * dAdZ
