import numpy as np

from .optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(self, optim_params, lr, alpha, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.alpha = alpha
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.alpha * self.v[ind] + (1 - self.alpha) * (grad ** 2)
        return self.lr * grad / (np.sqrt(self.v[ind] + self.eps))
