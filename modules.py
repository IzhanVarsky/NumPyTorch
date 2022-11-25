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


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.x_norm = None
        self.sqrtvar = None
        self.xmu = None
        self.gamma = Variable(np.ones(num_features))
        self.beta = Variable(np.zeros(num_features))
        self.EX = 0
        self.VarX = 1
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.EX = self.EX * self.momentum + mean * (1 - self.momentum)
            self.VarX = self.VarX * self.momentum + var * (1 - self.momentum)
        else:
            mean = self.EX
            var = self.VarX

        self.xmu = x - mean
        self.sqrtvar = np.sqrt(var + self.eps)
        self.x_norm = self.xmu / self.sqrtvar
        return self.x_norm * self.gamma.value + self.beta.value

    def backward(self, grad):
        N = grad.shape[0]
        self.beta.grad += grad.sum(axis=0)
        self.gamma.grad += (grad * self.x_norm).sum(axis=0)
        dxhat = grad * self.gamma.value
        divar = (dxhat * self.xmu).sum(axis=0)
        dx1 = dxhat / self.sqrtvar - \
              self.xmu / N * np.ones(grad.shape) * divar / (self.sqrtvar ** 3)
        dx2 = -np.ones(grad.shape) * dx1.sum(axis=0) / N
        return dx1 + dx2

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.x_norm = None
        self.sqrtvar = None
        self.xmu = None
        self.gamma = Variable(np.ones((1, num_features, 1, 1)))
        self.beta = Variable(np.zeros((1, num_features, 1, 1)))
        self.EX = 0
        self.VarX = 1
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = x.var(axis=(0, 2, 3), keepdims=True)
            self.EX = self.EX * self.momentum + mean * (1 - self.momentum)
            self.VarX = self.VarX * self.momentum + var * (1 - self.momentum)
        else:
            mean = self.EX
            var = self.VarX

        self.xmu = x - mean
        self.sqrtvar = np.sqrt(var + self.eps)
        self.x_norm = self.xmu / self.sqrtvar
        return self.x_norm * self.gamma.value + self.beta.value

    def backward(self, grad):
        N = grad.shape[0] * grad.shape[2] * grad.shape[3]
        self.beta.grad += grad.sum(axis=(0, 2, 3), keepdims=True)
        self.gamma.grad += (grad * self.x_norm).sum(axis=(0, 2, 3), keepdims=True)
        dxhat = grad * self.gamma.value
        divar = (dxhat * self.xmu).sum(axis=(0, 2, 3), keepdims=True)
        dx1 = dxhat / self.sqrtvar - self.xmu * divar / (self.sqrtvar ** 3) / N
        res = dx1 - dx1.sum(axis=(0, 2, 3), keepdims=True) / N
        return res

    def parameters(self):
        return [self.gamma, self.beta]
