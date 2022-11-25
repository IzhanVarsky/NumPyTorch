import numpy as np

from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, optim_params, lr, alpha=0.9, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.alpha = alpha
        self.m = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        return self.step_original(ind, grad)
        # return self.step_by_torch(ind, grad)

    def step_original(self, ind, grad):
        self.m[ind] = self.alpha * self.m[ind] + self.lr * grad
        return self.m[ind]

    def step_by_torch(self, ind, grad):
        self.m[ind] = self.alpha * self.m[ind] + (1 - self.alpha) * grad
        return self.lr * self.m[ind]
