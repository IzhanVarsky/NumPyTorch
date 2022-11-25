import numpy as np

from .module import Module
from .variable import Variable


class Linear(Module):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        self.old_x = None

        self.weights = Variable(np.random.randn(n_in, n_out) * np.sqrt(2 / n_in))
        self.biases = Variable(np.zeros(n_out))

    def forward(self, x):
        self.old_x = x
        return x @ self.weights.value + self.biases.value

    def backward(self, grad):
        self.biases.grad += grad.mean(axis=0)
        self.weights.grad += (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)
        return grad @ self.weights.value.T

    def parameters(self):
        return [self.weights, self.biases]
