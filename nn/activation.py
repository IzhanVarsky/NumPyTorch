import numpy as np

from .module import Module


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.old_x = None

    def forward(self, x):
        self.old_x = x
        return x.clip(min=0, max=None)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.old_y = None

    def forward(self, x):
        self.old_y = 1.0 / (1.0 + np.exp(-x))
        return self.old_y

    def backward(self, grad):
        return self.old_y * (1.0 - self.old_y) * grad


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.old_ch = None

    def forward(self, x):
        e1 = np.exp(x)
        e2 = np.exp(-x)
        s = e1 + e2
        self.old_ch = s / 2.0
        return (e1 - e2) / s

    def backward(self, grad):
        return grad / (self.old_ch ** 2)


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.old_y = None

    def forward(self, x):
        self.old_y = np.exp(x) / np.exp(x).sum(axis=1)[:, None]
        return self.old_y

    def backward(self, grad):
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])
