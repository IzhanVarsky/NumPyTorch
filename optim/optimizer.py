from abc import abstractmethod, ABC


class Optimizer(ABC):
    def __init__(self, optim_params, lr, maximize=False):
        self.optim_params = optim_params
        self.iter = 1
        self.lr = lr
        self.maximize = maximize

    def step(self):
        for ind, param in enumerate(self.optim_params):
            step = self.get_grad_step(ind, param.grad)
            if self.maximize:
                param.value += step
            else:
                param.value -= step
        self.iter += 1

    @abstractmethod
    def get_grad_step(self, ind, grad):
        raise NotImplementedError()

    def zero_grad(self):
        for ind, param in enumerate(self.optim_params):
            param.grad *= 0
