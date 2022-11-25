import numpy as np


class Variable:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
