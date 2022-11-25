from .loss import Loss
from .module import Module


class Net(Module):
    def __init__(self, module: Module, cost: Loss):
        super().__init__()
        self.module = module
        self.cost = cost

    def forward(self, x):
        return self.module(x)

    def loss(self, *args):
        return self.cost(*args)

    def backward(self):
        grad = self.cost.backward()
        self.module.backward(grad)

    def parameters(self):
        return self.module.parameters()

    def train(self, flag=True):
        self.module.train(flag)
