import numpy as np

constant_lr = lambda lr0: (lambda _: lr0)
linear_lr = lambda lr0, k: (lambda t: lr0 - k * t)
time_based_lr = lambda lr0, k: (lambda t: lr0 / (1 + k * t))
expo1_lr = lambda lr0, k: (lambda t: lr0 / (k ** t))


class AbstractOptimizer:
    def __init__(self, optim_params, lr, scheduler=constant_lr, start_iter=0):
        self.optim_params = optim_params
        self.iter = start_iter
        self.lr = scheduler(lr)

    def step(self):
        for ind, param in enumerate(self.optim_params):
            param.value -= self.get_grad_step(ind, param.grad)

    def get_grad_step(self, ind, grad):
        pass

    def zero_grad(self):
        for ind, param in enumerate(self.optim_params):
            param.grad *= 0

    def scheduler_step(self):
        self.iter += 1


class SGD(AbstractOptimizer):
    def __init__(self, optim_params, lr, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)

    def get_grad_step(self, ind, grad):
        return self.lr(self.iter) * grad


class Momentum(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha=0.9, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.last_grad = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        return self.step1(ind, grad)
        # return self.step2(ind, grad)

    def step1(self, ind, grad):
        self.last_grad[ind] = self.alpha * self.last_grad[ind] + self.lr(self.iter) * grad
        return self.last_grad[ind]

    def step2(self, ind, grad):
        self.last_grad[ind] = self.alpha * self.last_grad[ind] + (1 - self.alpha) * grad
        return self.lr(self.iter) * self.last_grad[ind]


class Nesterov(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha: float, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.last_grad = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        momentum = self.alpha * self.last_grad[ind]
        # TODO: How to do??
        # self.last_grad[ind] = momentum + self.lr(iter) * calc_grad(theta - momentum)
        return self.last_grad[ind]


class Adagrad(AbstractOptimizer):
    def __init__(self, optim_params, lr, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.v[ind] + grad ** 2
        return self.lr(self.iter) / ((self.v[ind] + self.eps) ** 0.5) * grad


class RMSProp(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.alpha * self.v[ind] + (1 - self.alpha) * (grad ** 2)
        return self.lr(self.iter) / ((self.v[ind] + self.eps) ** 0.5) * grad


class Adam(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha1=0.9, alpha2=0.999, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.last_grad = np.zeros_like(optim_params)
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        iter = self.iter
        self.last_grad[ind] = self.alpha1 * self.last_grad[ind] + (1 - self.alpha1) * grad
        self.v[ind] = self.alpha2 * self.v[ind] + (1 - self.alpha2) * (grad ** 2)
        last_grad_norm = self.last_grad[ind] / (1 - self.alpha1 ** (iter + 1))
        v_norm = self.v[ind] / (1 - self.alpha2 ** (iter + 1))
        return self.lr(iter) / ((v_norm + self.eps) ** 0.5) * last_grad_norm
