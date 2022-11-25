import collections
from itertools import repeat

import numpy as np

from .module import Module
from .variable import Variable


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2d(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # H x W
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # in TORCH: OUT x IN x H * W
        self.weights = Variable(np.random.random((out_channels, in_channels,
                                                  self.kernel_size[0], self.kernel_size[1])) * 0.1)
        self.biases = Variable(np.random.random(out_channels) * 0.01)
        self.old_img = None
        self.padded_img_shape = None

    def forward(self, x):
        bz, C, H, W = x.shape
        pad_shape = ((0, 0), (0, 0), self.padding, self.padding)
        x = np.pad(x, pad_shape, constant_values=(0, 0))

        h_stride, w_stride = self.stride
        h_dilation, w_dilation = self.dilation
        h_ker, w_ker = self.kernel_size

        h_in = H + 2 * self.padding[0]
        w_in = W + 2 * self.padding[1]
        self.padded_img_shape = (h_in, w_in)

        h_out = (h_in - h_dilation * (self.kernel_size[0] - 1) - 1) // h_stride + 1
        w_out = (w_in - w_dilation * (self.kernel_size[1] - 1) - 1) // w_stride + 1

        self.old_img = np.zeros((bz, self.out_channels, self.in_channels,
                                 h_ker, w_ker, h_out, w_out))

        mega_res = np.zeros((bz, self.out_channels, h_out, w_out), dtype=np.float32)
        for b_ind in range(bz):
            for out_ind in range(self.out_channels):
                for c_ind in range(C):
                    ker = self.weights.value[out_ind][c_ind]
                    img = x[b_ind][c_ind]
                    for i in range(h_out):
                        for j in range(w_out):
                            for a in range(h_ker):
                                for b in range(w_ker):
                                    img_h_pos = i * h_stride + a * h_dilation
                                    img_w_pos = j * w_stride + b * w_dilation
                                    mega_res[b_ind][out_ind][i][j] += img[img_h_pos][img_w_pos] * ker[a][b]
                                    self.old_img[b_ind][out_ind][c_ind][a][b][i][j] = img[img_h_pos][img_w_pos]
                mega_res[b_ind][out_ind] += self.biases.value[out_ind]
        return mega_res

    def backward(self, grad):
        bz = grad.shape[0]
        h_stride, w_stride = self.stride
        h_pad, w_pad = self.padding
        h_ker, w_ker = self.kernel_size

        self.biases.grad += grad.mean(axis=0).sum(axis=(-2, -1))

        total_res = np.zeros((bz, self.in_channels,
                              self.padded_img_shape[0] - 2 * h_pad,
                              self.padded_img_shape[1] - 2 * w_pad), dtype=np.float32)
        self.weights.grad = (self.old_img.transpose((2, 3, 4, 1, 0, 5, 6)) * grad.transpose((1, 0, 2, 3))) \
            .sum(axis=(-1, -2, -3)).transpose(3, 0, 1, 2)
        for b_ind in range(bz):
            for j in range(self.out_channels):
                # self.weights.grad[j] += (self.old_img[b_ind][j] * grad[b_ind][j]).sum(axis=(-2, -1))
                cur_grad = grad[b_ind][j]
                h_out, w_out = cur_grad.shape
                for i in range(self.in_channels):
                    for i1 in range(h_pad, self.padded_img_shape[0] - h_pad):
                        for j1 in range(w_pad, self.padded_img_shape[1] - w_pad):
                            for i_x_l in range(h_out):
                                # for i_x_l in range((i1 - h_ker + 1) // h_stride, min(h_out, i1 // h_stride + 1)):
                                for j_x_l in range(w_out):
                                    # TODO: Is division to dilation needed?
                                    a = i1 - i_x_l * h_stride
                                    b = j1 - j_x_l * w_stride
                                    if 0 <= a < h_ker and 0 <= b < w_ker:
                                        total_res[b_ind][i][i1 - h_pad][j1 - w_pad] += cur_grad[i_x_l][j_x_l] * \
                                                                                       self.weights.value[j][i][a][b]
        self.weights.grad /= bz
        return total_res

    def parameters(self):
        return [self.weights, self.biases]
