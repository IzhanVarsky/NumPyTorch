from abc import ABC, abstractmethod
import numpy as np
from .module import Module


class Loss(Module, ABC):
    @abstractmethod
    def backward(self):
        raise NotImplementedError()


class CrossEntropyLoss(Loss):
    def __init__(self, label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__()
        # TODO: add reduction policy
        self.label_smoothing = label_smoothing
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        # TODO: add check if target is class labels or probs
        bz, k = x.shape
        self.cached_input = x.clip(min=1e-8, max=None)  # to avoid division by zero
        self.cached_target = target * (1 - self.label_smoothing) + self.label_smoothing / k
        return -(self.cached_target * np.log(self.cached_input)).sum(axis=1)

    def backward(self):
        return -self.cached_target / self.cached_input / self.cached_input.shape[0]


class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        self.cached_input = x
        self.cached_target = target
        return ((target - x) ** 2).mean(axis=1)

    def backward(self):
        return 2 * (self.cached_input - self.cached_target) / np.prod(self.cached_input.shape)
