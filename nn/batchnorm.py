import numpy as np

from .module import Module
from .variable import Variable


def _shape_without_one_axis(a, axis=1):
    axes = np.arange(a.ndim)
    axes = np.delete(axes, axis)
    return tuple(axes)


class BaseBatchNorm(Module):
    def __init__(self, used_shape, eps=1e-5, momentum=0.1):
        super(BaseBatchNorm, self).__init__()
        self.x_norm = None
        self.sqrtvar = None
        self.xmu = None
        self.EX = 0
        self.VarX = 1
        self.eps = eps
        self.momentum = momentum
        self.gamma = Variable(np.ones(used_shape))
        self.beta = Variable(np.zeros(used_shape))

    def forward(self, x):
        axes = _shape_without_one_axis(x)

        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True)
            self.EX = self.EX * self.momentum + mean * (1 - self.momentum)
            self.VarX = self.VarX * self.momentum + var * (1 - self.momentum)
        else:
            mean = self.EX
            var = self.VarX

        self.xmu = x - mean
        self.sqrtvar = np.sqrt(var + self.eps)
        self.x_norm = self.xmu / self.sqrtvar
        return self.x_norm * self.gamma.data + self.beta.data

    def backward(self, grad):
        axes = _shape_without_one_axis(grad)

        N = np.prod(np.delete(grad.shape, 1))
        self.beta.grad += grad.sum(axis=axes, keepdims=True)
        self.gamma.grad += (grad * self.x_norm).sum(axis=axes, keepdims=True)
        dxhat = grad * self.gamma.data
        divar = (dxhat * self.xmu).sum(axis=axes, keepdims=True)
        dx1 = dxhat / self.sqrtvar - self.xmu * divar / (self.sqrtvar ** 3) / N
        res = dx1 - dx1.sum(axis=axes, keepdims=True) / N
        return res

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm1d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__((1, num_features), eps, momentum)


class BatchNorm2d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__((1, num_features, 1, 1), eps, momentum)
