# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class AdamW:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay = weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # Calculate updates for weight
            self.m_W[layer_id] = self.m_W[layer_id] * self.beta1 + layer.dLdW * (1 - self.beta1)
            self.v_W[layer_id] = self.v_W[layer_id] * self.beta2 + layer.dLdW**2 * (1 - self.beta2)

            # Calculate updates for bias
            self.m_b[layer_id] = self.m_b[layer_id] * self.beta1 + layer.dLdb * (1 - self.beta1)
            self.v_b[layer_id] = self.v_b[layer_id] * self.beta2 + layer.dLdb**2 * (1 - self.beta2)

            # Perform weight and bias updates
            unbiased_m_W = self.m_W[layer_id] / (1 - self.beta1**self.t)
            unbiased_v_W = self.v_W[layer_id] / (1 - self.beta2**self.t)
            unbiased_m_b = self.m_b[layer_id] / (1 - self.beta1**self.t)
            unbiased_v_b = self.v_b[layer_id] / (1 - self.beta2**self.t)

            layer.W -= self.lr * unbiased_m_W / np.sqrt(self.eps + unbiased_v_W)
            layer.b -= self.lr * unbiased_m_b / np.sqrt(self.eps + unbiased_v_b)

            # Perform weight decay
            layer.W *= 1 - self.weight_decay * self.lr
            layer.b *= 1 - self.weight_decay * self.lr
