import numpy as np

from .module import Module
from .variable import Variable


class Linear(Module):
    def __init__(self, in_features, out_features, dtype=np.float32):
        super(Linear, self).__init__()
        self.weight = Variable(np.random.randn(out_features, in_features).astype(dtype) * np.sqrt(2 / in_features))
        self.bias = Variable(np.zeros(out_features, dtype=dtype))
        self.cached_x = None

    def forward(self, x):
        self.cached_x = x
        return x @ self.weight.data.T + self.bias.data

    def backward(self, grad):
        self.bias.grad += grad.sum(axis=0)
        self.weight.grad += (self.cached_x[:, None, :] * grad[:, :, None]).sum(axis=0)
        return grad @ self.weight.data

    def parameters(self):
        return [self.weight, self.bias]
