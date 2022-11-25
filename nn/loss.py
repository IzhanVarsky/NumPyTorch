from abc import ABC, abstractmethod
import numpy as np
from .module import Module


class Loss(Module, ABC):
    @abstractmethod
    def backward(self):
        raise NotImplementedError()


class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.old_x = None
        self.old_y = None

    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)  # to avoid division by zero
        self.old_y = y
        return (np.where(y == 1, -np.log(self.old_x), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.old_y == 1, -1 / self.old_x, 0)
