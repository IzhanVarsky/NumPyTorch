import numpy as np

from .optimizer import Optimizer


class Adagrad(Optimizer):
    def __init__(self, optim_params, lr, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.v[ind] + grad ** 2
        return self.lr / ((self.v[ind] + self.eps) ** 0.5) * grad
