import numpy as np

from .module import Module
from .variable import Variable


class Linear(Module):
    def __init__(self, in_features, out_features, dtype=np.float32):
        super(Linear, self).__init__()
        self.old_x = None

        self.weights = Variable(np.random.randn(out_features, in_features).astype(dtype) * np.sqrt(2 / in_features))
        self.biases = Variable(np.zeros(out_features, dtype=dtype))

    def forward(self, x):
        self.old_x = x
        return x @ self.weights.value.T + self.biases.value

    def backward(self, grad):
        self.biases.grad += grad.sum(axis=0)
        self.weights.grad += (self.old_x[:, None, :] * grad[:, :, None]).sum(axis=0)
        return grad @ self.weights.value

    def parameters(self):
        return [self.weights, self.biases]
