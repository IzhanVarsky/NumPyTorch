from .module import Module


class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()
        self.cached_x_shape = None

    def forward(self, x):
        self.cached_x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.cached_x_shape)
