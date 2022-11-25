from .optimizer import Optimizer


class SGD(Optimizer):
    def get_grad_step(self, ind, grad):
        return self.lr * grad
