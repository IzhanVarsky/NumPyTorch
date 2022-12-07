from abc import ABC, abstractmethod
import numpy as np
from .module import Module


class Loss(Module, ABC):
    def __init__(self, reduction: str = 'mean'):
        super(Loss, self).__init__()
        possible_reductions = ['none', 'mean', 'sum']
        if reduction not in possible_reductions:
            print(f"Warning: `{reduction}` reduction is unknown. "
                  f"`mean` reduction will be used instead.")
            reduction = 'mean'
        self.reduction = reduction

    @abstractmethod
    def backward(self):
        raise NotImplementedError()


class CrossEntropyLoss(Loss):
    def __init__(self, reduction: str = 'mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__(reduction)
        self.label_smoothing = label_smoothing
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        # TODO: add check if target is class labels or probs
        bz, k = x.shape
        self.cached_input = x.clip(min=1e-8, max=None)  # to avoid division by zero
        self.cached_target = target * (1 - self.label_smoothing) + self.label_smoothing / k
        out = -(self.cached_target * np.log(self.cached_input)).sum(axis=1)
        if self.reduction == 'mean':
            return out.mean()
        if self.reduction == 'sum':
            return out.sum()
        return out

    def backward(self):
        reduction = self.reduction
        if self.reduction == 'none':
            print(f"Warning: backward for `none` reduction is not defined. "
                  f"`mean` reduction is used instead.")
            reduction = 'mean'
        out = -self.cached_target / self.cached_input
        if reduction == 'mean':
            return out / self.cached_input.shape[0]
        if reduction == 'sum':
            return out
        return out


class MSELoss(Loss):
    def __init__(self, reduction: str = 'mean'):
        super(MSELoss, self).__init__(reduction)
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        self.cached_input = x
        self.cached_target = target
        out = ((target - x) ** 2)
        if self.reduction == 'mean':
            return out.mean()
        if self.reduction == 'sum':
            return out.sum()
        return out

    def backward(self):
        reduction = self.reduction
        if self.reduction == 'none':
            print(f"Warning: backward for `none` reduction is not defined. "
                  f"`mean` reduction is used instead.")
            reduction = 'mean'
        out = 2 * (self.cached_input - self.cached_target)
        if reduction == 'mean':
            return out / np.prod(self.cached_input.shape)
        if reduction == 'sum':
            return out
        return out
