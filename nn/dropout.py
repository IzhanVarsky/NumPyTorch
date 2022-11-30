import numpy as np

from .module import Module


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be in range [0, 1], but got {p}")
        self.p = p  # probability of an element to be zeroed
        self.cached_mask = None

    def forward(self, x):
        if self.training:
            self.cached_mask = (np.random.rand(*x.shape) > self.p)
            return self.inverted_dropout(x, self.cached_mask)
        return x

    def backward(self, grad):
        if self.training:
            return self.inverted_dropout(grad, self.cached_mask)
        return grad

    def inverted_dropout(self, x, mask):
        x *= mask
        return x / (1 - self.p)
