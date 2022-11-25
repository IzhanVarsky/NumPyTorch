from abc import ABC, abstractmethod

from .optimizer import Optimizer


class SchedulerLR(ABC):
    def __init__(self, optim: Optimizer):
        self.optim = optim
        self.start_lr = optim.lr
        self.cur_step_num = 1

    def step(self):
        self.optim.lr = self.get_new_lr()
        self.cur_step_num += 1

    @abstractmethod
    def get_new_lr(self):
        raise NotImplementedError()


class ConstantLR(SchedulerLR):
    def get_new_lr(self):
        return self.start_lr


class LinearLR(SchedulerLR):
    def __init__(self, optim, k):
        super().__init__(optim)
        self.k = k

    def get_new_lr(self):
        return self.start_lr - self.k * self.cur_step_num


class TimeBasedLR(SchedulerLR):
    def __init__(self, optim, k):
        super().__init__(optim)
        self.k = k

    def get_new_lr(self):
        return self.start_lr / (1 + self.k * self.cur_step_num)


class ExpoLR(SchedulerLR):
    def __init__(self, optim, k):
        super().__init__(optim)
        self.k = k

    def get_new_lr(self):
        return self.start_lr / (self.k ** self.cur_step_num)
