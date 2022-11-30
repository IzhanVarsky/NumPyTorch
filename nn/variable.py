import numpy as np


class Variable:
    def __init__(self, data, dtype=np.float32):
        self.data = data.astype(dtype)
        self.grad = np.zeros_like(data, dtype=dtype)
