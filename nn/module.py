from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        return []

    def eval(self):
        self.train(False)

    def train(self, flag=True):
        self.training = flag
