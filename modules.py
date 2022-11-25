import numpy as np


class Module:
    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def parameters(self):
        return []

    def eval(self):
        self.train(False)

    def train(self, flag=True):
        self.training = flag


class Variable:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class CrossEntropy(Module):
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


class Linear(Module):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
        self.old_x = None

        self.weights = Variable(np.random.randn(n_in, n_out) * np.sqrt(2 / n_in))
        self.biases = Variable(np.zeros(n_out))

    def forward(self, x):
        self.old_x = x
        return x @ self.weights.value + self.biases.value

    def backward(self, grad):
        self.biases.grad += grad.mean(axis=0)
        self.weights.grad += (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)
        return grad @ self.weights.value.T

    def parameters(self):
        return [self.weights, self.biases]


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.x_shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to in range [0, 1], but got {p}")
        self.p = p  # probability of an element to be zeroed
        self.old_mask = None

    def forward(self, x):
        if self.training:
            self.old_mask = (np.random.rand(*x.shape) > self.p)
            return self.inverted_dropout(x, self.old_mask)
        return x

    def backward(self, grad):
        if self.training:
            return self.inverted_dropout(grad, self.old_mask)
        return grad

    def inverted_dropout(self, x, mask):
        x *= mask
        return x / (1 - self.p)
