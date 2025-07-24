import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))  # Shape (C1 ,C0)
        self.b = np.zeros((out_features, 1))  # Shape (C1, 1)
        self.dLdW = np.zeros((out_features, in_features))  # Shape (C1, C0)
        self.dLdb = np.zeros((out_features, 1))  # Shape (C1, 1)

        self.in_features = in_features
        self.out_features = out_features

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1))  # Shape (N, 1)
        Z = self.A @ self.W.T + self.Ones @ self.b.T  # Shape (N, C1)

        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T  # Shape (C0, C1)
        dZdW = self.A  # Shape (N, C0)
        dZdb = self.Ones  # Shape (N, 1)

        # dLdZ Shape (N, C1)
        dLdA = dLdZ @ dZdA.T  # Shape (N, C0)
        dLdW = dLdZ.T @ dZdW  # Shape (C1, C0)
        dLdb = dLdZ.T @ dZdb  # Shape (C1, 1)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
