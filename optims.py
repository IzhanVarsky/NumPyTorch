import numpy as np

constant_lr = lambda lr0: (lambda _: lr0)
linear_lr = lambda k: lambda lr0: (lambda t: lr0 - k * t)
time_based_lr = lambda k: lambda lr0: (lambda t: lr0 / (1 + k * t))
expo1_lr = lambda k: lambda lr0: (lambda t: lr0 / (k ** t))


class AbstractOptimizer:
    def __init__(self, optim_params, lr, scheduler=constant_lr, maximize=False):
        self.optim_params = optim_params
        self.iter = 1
        self.scheduler_iter = 0
        self.lr = scheduler(lr)
        self.maximize = maximize

    def step(self):
        for ind, param in enumerate(self.optim_params):
            step = self.get_grad_step(ind, param.grad)
            if self.maximize:
                param.value += step
            else:
                param.value -= step
        self.iter += 1

    def get_grad_step(self, ind, grad):
        pass

    def zero_grad(self):
        for ind, param in enumerate(self.optim_params):
            param.grad *= 0

    def scheduler_step(self):
        self.scheduler_iter += 1


class SGD(AbstractOptimizer):
    def __init__(self, optim_params, lr, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)

    def get_grad_step(self, ind, grad):
        return self.lr(self.scheduler_iter) * grad


class Momentum(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha=0.9, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.m = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        return self.step_original(ind, grad)
        # return self.step_by_torch(ind, grad)

    def step_original(self, ind, grad):
        self.m[ind] = self.alpha * self.m[ind] + self.lr(self.scheduler_iter) * grad
        return self.m[ind]

    def step_by_torch(self, ind, grad):
        self.m[ind] = self.alpha * self.m[ind] + (1 - self.alpha) * grad
        return self.lr(self.scheduler_iter) * self.m[ind]


class Nesterov(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha: float, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.last_grad = np.zeros_like(optim_params)

    def get_grad_step(self, ind, grad):
        momentum = self.alpha * self.last_grad[ind]
        # TODO: How to do??
        # self.last_grad[ind] = momentum + self.lr(self.scheduler_iter) * calc_grad(theta - momentum)
        return self.last_grad[ind]


class Adagrad(AbstractOptimizer):
    def __init__(self, optim_params, lr, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.v[ind] + grad ** 2
        return self.lr(self.scheduler_iter) / ((self.v[ind] + self.eps) ** 0.5) * grad


class RMSProp(AbstractOptimizer):
    def __init__(self, optim_params, lr, alpha, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
        self.alpha = alpha
        self.v = np.zeros_like(optim_params)
        self.eps = 1e-8

    def get_grad_step(self, ind, grad):
        self.v[ind] = self.alpha * self.v[ind] + (1 - self.alpha) * (grad ** 2)
        return self.lr(self.scheduler_iter) * grad / (np.sqrt(self.v[ind] + self.eps))


class Adam(AbstractOptimizer):
    def __init__(self, optim_params, lr, beta1=0.9, beta2=0.999, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
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
        return self.lr(self.scheduler_iter) / ((v_norm + self.eps) ** 0.5) * m_norm


class RAdam(AbstractOptimizer):
    def __init__(self, optim_params, lr, beta1=0.9, beta2=0.999, scheduler=constant_lr):
        super().__init__(optim_params, lr, scheduler)
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
        return self.lr(self.scheduler_iter) * m_norm * c
