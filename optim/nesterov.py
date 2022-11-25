import numpy as np

from .optimizer import Optimizer


class Nesterov(Optimizer):
    def __init__(self, optim_params, lr, alpha: float, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.alpha = alpha
        self.last_grad = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        # momentum = self.alpha * self.last_grad[ind]
        # TODO: How to do??
        # self.last_grad[ind] = momentum + self.lr * calc_grad(theta - momentum)
        # return self.last_grad[ind]
        raise NotImplementedError()
