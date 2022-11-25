import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, optim_params, lr, beta1=0.9, beta2=0.999, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros_like(optim_params)
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.m[ind] = self.beta1 * self.m[ind] + (1 - self.beta1) * grad
        self.v[ind] = self.beta2 * self.v[ind] + (1 - self.beta2) * (grad ** 2)
        m_norm = self.m[ind] / (1 - self.beta1 ** self.iter)
        v_norm = self.v[ind] / (1 - self.beta2 ** self.iter)
        return self.lr / ((v_norm + self.eps) ** 0.5) * m_norm
