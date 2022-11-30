import numpy as np

from .module import Module
from .utils import _pair


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.anti_cached = None
        self.old_shape = None
        self.cached_indices = None
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if (stride is not None) else kernel_size
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, x):
        bz, C, H, W = x.shape
        self.old_shape = x.shape
        h_pad, w_pad = self.padding
        pad_shape = ((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad))
        x = np.pad(x, pad_shape, constant_values=-np.inf)

        h_stride, w_stride = self.stride
        h_dilation, w_dilation = self.dilation
        h_ker, w_ker = self.kernel_size

        h_in, w_in = x.shape[-2:]
        h_out = (h_in - h_dilation * (h_ker - 1) - 1) // h_stride + 1
        w_out = (w_in - w_dilation * (w_ker - 1) - 1) // w_stride + 1

        res = np.full((bz, C, h_out, w_out), -np.inf, dtype=np.float32)
        self.cached_indices = np.empty((bz, C, h_out, w_out, 2), dtype=np.int)
        self.anti_cached = np.empty((bz, C), dtype=dict)
        self.anti_cached_old = {}
        for b_ind in range(bz):
            for c_ind in range(C):
                img = x[b_ind][c_ind]
                self.anti_cached[b_ind][c_ind] = {}
                for i in range(h_out):
                    for j in range(w_out):
                        h_positions = np.arange(h_ker) * h_dilation + i * h_stride
                        w_positions = np.arange(w_ker) * w_dilation + j * w_stride
                        img_filtered = img[h_positions.reshape(-1, 1), w_positions]
                        M = img_filtered.argmax()
                        h1, w1 = M // w_ker, M % w_ker
                        i_back = h_positions[h1] - h_pad
                        j_back = w_positions[w1] - w_pad
                        res[b_ind][c_ind][i][j] = img_filtered[h1][w1]
                        key = (i_back, j_back)
                        back = [i, j]
                        if key in self.anti_cached[b_ind][c_ind]:
                            self.anti_cached[b_ind][c_ind][key].append(back)
                        else:
                            self.anti_cached[b_ind][c_ind][key] = [back]
                        # back2 = [b_ind, c_ind, i, j]
                        # if key in self.anti_cached_old:
                        #     self.anti_cached_old[key].append(back2)
                        # else:
                        #     self.anti_cached_old[key] = [back2]
                        # self.cached_indices[b_ind][c_ind][i][j] = [i_back, j_back]

                        # for a in range(h_ker):
                        #     for b in range(w_ker):
                        #         img_h_pos = i * h_stride + a * h_dilation
                        #         img_w_pos = j * w_stride + b * w_dilation
                        #         if 0 <= img_h_pos < h_in and 0 <= img_w_pos < w_in:
                        #             if img[img_h_pos][img_w_pos] > res[b_ind][c_ind][i][j]:
                        #                 # subtract self.padding from coords to make
                        #                 # backprop correct for padded matrices
                        #                 i_back = img_h_pos - self.padding[0]
                        #                 j_back = img_w_pos - self.padding[1]
                        #                 res[b_ind][c_ind][i][j] = img[img_h_pos][img_w_pos]
                        #                 self.cached_indices[b_ind][c_ind][i][j] = [i_back, j_back]
        return res

    def backward(self, grad):
        bz, C, H, W = grad.shape
        total_res1 = np.zeros(self.old_shape, dtype=np.float32)
        total_res2 = np.zeros(self.old_shape, dtype=np.float32)
        total_res3 = np.zeros(self.old_shape, dtype=np.float32)
        # for key in self.anti_cached_old:
        #     i_back, j_back = key
        #     for (b_ind, c_ind, i, j) in self.anti_cached_old[key]:
        #         total_res3[b_ind][c_ind][i_back][j_back] += grad[b_ind][c_ind][i][j]
        for b_ind in range(bz):
            for c_ind in range(C):
                cur_anti_cached = self.anti_cached[b_ind][c_ind]
                for key in cur_anti_cached:
                    i_back, j_back = key
                    ab_indices = np.array(cur_anti_cached[key])
                    total_res1[b_ind][c_ind][i_back][j_back] += \
                        grad[b_ind][c_ind][ab_indices[:, 0], ab_indices[:, 1]].sum()
                    # for (i, j) in cur_anti_cached[key]:
                    #     total_res1[b_ind][c_ind][i_back][j_back] += grad[b_ind][c_ind][i][j]
                # for i in range(H):
                #     for j in range(W):
                #         i_back, j_back = self.cached_indices[b_ind][c_ind][i][j]
                #         total_res2[b_ind][c_ind][i_back][j_back] += grad[b_ind][c_ind][i][j]
        # if (total_res2 != total_res1).sum() > 0:
        # raise NotImplementedError()
        # print((np.absolute(total_res2) - np.absolute(total_res1)).sum())
        return total_res1
