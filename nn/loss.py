from abc import ABC, abstractmethod
import numpy as np
from .module import Module
from .utils import one_hot


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
    def __init__(self, weight=None, reduction: str = 'mean', label_smoothing=0.0):
        super(CrossEntropyLoss, self).__init__(reduction)
        self.weight = 1.0 if weight is None else weight
        self.label_smoothing = label_smoothing
        self.cached_input = None
        self.cached_target = None
        self.norm = None

    def forward(self, x, target):
        k = x.shape[-1]
        self.norm = self.get_batch_size(x)
        if target.dtype.kind in "iu":
            # Target contains class indices
            self.norm = self.weight[target].sum()
            target = one_hot(target, k)
        self.cached_input = x.clip(min=1e-8, max=None)  # to avoid division by zero
        self.cached_target = target * (1 - self.label_smoothing) + self.label_smoothing / k
        out = -(self.weight * self.cached_target * np.log(self.cached_input)).sum(axis=-1)
        if self.reduction == 'mean':
            return out.sum() / self.norm
        if self.reduction == 'sum':
            return out.sum()
        return out

    def backward(self):
        reduction = self.reduction
        if self.reduction == 'none':
            print(f"Warning: backward for `none` reduction is not defined. "
                  f"`mean` reduction is used instead.")
            reduction = 'mean'
        out = -self.weight * self.cached_target / self.cached_input
        if reduction == 'mean':
            return out / self.norm
        if reduction == 'sum':
            return out
        return out

    @staticmethod
    def get_batch_size(arr):
        if arr.ndim == 1:
            return 1
        return arr.shape[0]


class MSELoss(Loss):
    def __init__(self, reduction: str = 'mean'):
        super(MSELoss, self).__init__(reduction)
        self.cached_input = None
        self.cached_target = None

    def forward(self, x, target):
        self.cached_input = x
        self.cached_target = target
        out = (target - x) ** 2
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
