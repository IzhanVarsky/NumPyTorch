import numpy as np

from .optimizer import Optimizer


class RAdam(Optimizer):
    def __init__(self, optim_params, lr, beta1=0.9, beta2=0.999, maximize=False):
        super().__init__(optim_params, lr, maximize)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros_like(optim_params)
        self.v = np.zeros_like(optim_params)
        self.p_inf = 2 / (1 - beta2) - 1
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.m[ind] = self.beta1 * self.m[ind] + (1 - self.beta1) * grad
        self.v[ind] = self.beta2 * self.v[ind] + (1 - self.beta2) * (grad ** 2)
        m_norm = self.m[ind] / (1 - self.beta1 ** self.iter)
        beta_pow_t = self.beta2 ** self.iter
        p = self.p_inf - 2 * self.iter * beta_pow_t / (1 - beta_pow_t)
        c = 1
        if p > 5:
            el = np.sqrt((1 - beta_pow_t) / (self.v[ind] + self.eps))
            r = np.sqrt((p - 4) * (p - 2) * self.p_inf / (self.p_inf - 4) / (self.p_inf - 2) / p)
            c = r * el
        return self.lr * m_norm * c
