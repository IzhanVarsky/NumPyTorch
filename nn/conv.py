import numpy as np
from .utils import _pair, get_all_selected_patches
from .module import Module
from .variable import Variable


class Conv2d(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 dtype=np.float32):
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
                                                  self.kernel_size[0], self.kernel_size[1])).astype(dtype) * 0.1)
        self.biases = Variable(np.random.random(out_channels).astype(dtype) * 0.01)
        self.old_img = None
        self.padded_img_shape = None

    def forward(self, x):
        bz, C, H, W = x.shape
        h_pad, w_pad = self.padding
        pad_shape = ((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad))
        x = np.pad(x, pad_shape, constant_values=(0, 0))

        h_stride, w_stride = self.stride
        h_dilation, w_dilation = self.dilation
        h_ker, w_ker = self.kernel_size

        h_in, w_in = x.shape[-2:]
        self.padded_img_shape = (h_in, w_in)

        h_out = (h_in - h_dilation * (h_ker - 1) - 1) // h_stride + 1
        w_out = (w_in - w_dilation * (w_ker - 1) - 1) // w_stride + 1

        self.old_img = np.zeros((bz, self.out_channels, self.in_channels,
                                 h_ker, w_ker, h_out, w_out))

        mega_res = np.zeros((bz, self.out_channels, h_out, w_out), dtype=np.float32)
        for b_ind in range(bz):
            for out_ind in range(self.out_channels):
                for c_ind in range(C):
                    ker = self.weights.data[out_ind][c_ind]
                    img = x[b_ind][c_ind]
                    for i in range(h_out):
                        for j in range(w_out):
                            for a in range(h_ker):
                                for b in range(w_ker):
                                    img_h_pos = i * h_stride + a * h_dilation
                                    img_w_pos = j * w_stride + b * w_dilation
                                    mega_res[b_ind][out_ind][i][j] += img[img_h_pos][img_w_pos] * ker[a][b]
                                    self.old_img[b_ind][out_ind][c_ind][a][b][i][j] = img[img_h_pos][img_w_pos]
                mega_res[b_ind][out_ind] += self.biases.data[out_ind]
        return mega_res

    def backward(self, grad):
        bz = grad.shape[0]
        h_stride, w_stride = self.stride
        h_pad, w_pad = self.padding
        h_ker, w_ker = self.kernel_size

        self.biases.grad += grad.mean(axis=0).sum(axis=(-2, -1))
        # self.old_img[b_ind][out_ind][c_ind][h_ker][w_ker][h_out][w_out]
        # grad[b_ind][out_ind][h_out][w_out]
        # self.select_img[bz, in_c, h_out, w_out, h_ker * w_ker]
        self.weights.grad = (grad[:, :, None, None, None, :, :] * self.old_img).sum(axis=(0, -2, -1)) / bz

        total_res = np.zeros((bz, self.in_channels,
                              self.padded_img_shape[0] - 2 * h_pad,
                              self.padded_img_shape[1] - 2 * w_pad), dtype=np.float32)
        for b_ind in range(bz):
            for j in range(self.out_channels):
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
                                                                                       self.weights.data[j][i][a][b]
        return total_res

    def parameters(self):
        return [self.weights, self.biases]


class FastConv2d(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 dtype=np.float32):
        super(FastConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # H x W
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        # in TORCH: OUT x IN x H * W
        self.weight = Variable(np.random.random((out_channels, in_channels, *self.kernel_size)).astype(dtype) * 0.1)
        self.bias = Variable(np.random.random(out_channels).astype(dtype) * 0.01)

    def forward(self, x):
        h_pad, w_pad = self.padding
        pad_shape = ((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad))
        x = np.pad(x, pad_shape, mode='constant')

        bz, C, h_in, w_in = x.shape
        self.padded_img_shape = (C, h_in, w_in)
        h_out = (h_in - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w_in - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        patches, i, j, k = get_all_selected_patches(x, self.kernel_size, self.stride, self.dilation, return_i_j_k=True)
        self.select_img = patches
        self.i = i
        self.j = j
        self.k = k
        convolve = self.weight.data[None, :, :, None, :, :] * patches[:, None]
        res = convolve.sum(axis=(2, -1, -2)).reshape(-1, self.out_channels, h_out, w_out)
        res = res + self.bias.data[None, :, None, None]
        return res

    def backward(self, grad):
        bz, out_c, H, W = grad.shape
        h_pad, w_pad = self.padding
        h_ker, w_ker = self.kernel_size

        self.bias.grad += grad.mean(axis=0).sum(axis=(-2, -1))

        in_c, out_h, out_w = self.padded_img_shape

        dw = grad[:, :, None, :, :, None, None] * self.select_img.reshape(bz, 1, in_c, H, W, h_ker, w_ker)
        self.weight.grad = dw.sum(axis=(0, 3, 4)) / bz

        X = grad[:, :, None, :, :, None, None] * self.weight.data.reshape(1, self.out_channels, in_c,
                                                                          1, 1, h_ker, w_ker)
        X = X.sum(axis=1)

        padded = np.zeros((bz, in_c, out_h, out_w))  # empty padded array
        X = X.transpose(0, 1, -2, -1, 2, 3)
        X = X.reshape(bz, h_ker * w_ker * in_c, -1)

        np.add.at(padded, (slice(None), self.k.reshape(-1, 1),
                           np.repeat(self.i, in_c).reshape(w_ker * h_ker * in_c, -1),
                           np.repeat(self.j, in_c).reshape(w_ker * h_ker * in_c, -1)), X)
        return padded[:, :, h_pad:out_h - h_pad, w_pad:out_w - w_pad]
