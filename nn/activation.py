import numpy as np

from .module import Module


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.cached_x = None

    def forward(self, x):
        self.cached_x = x
        return x.clip(min=0, max=None)

    def backward(self, grad):
        return np.where(self.cached_x > 0, grad, 0)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.cached_y = None

    def forward(self, x):
        self.cached_y = 1.0 / (1.0 + np.exp(-x))
        return self.cached_y

    def backward(self, grad):
        return self.cached_y * (1.0 - self.cached_y) * grad


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.cached_ch = None

    def forward(self, x):
        e1 = np.exp(x)
        e2 = np.exp(-x)
        s = e1 + e2
        self.cached_ch = s / 2.0
        return (e1 - e2) / s

    def backward(self, grad):
        return grad / (self.cached_ch ** 2)


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.cached_y = None

    def forward(self, x):
        self.cached_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        return self.cached_y

    def backward(self, grad):
        return self.cached_y * (grad - (grad * self.cached_y).sum(axis=1)[:, None])
