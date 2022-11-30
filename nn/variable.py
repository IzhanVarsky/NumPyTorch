import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)
