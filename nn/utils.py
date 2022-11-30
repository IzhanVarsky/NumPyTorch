import collections
from itertools import repeat

import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


def pad_img(x, padding, pad_value=0):
    h_pad, w_pad = padding
    pad_shape = ((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad))
    return np.pad(x, pad_shape, constant_values=pad_value)


def get_all_selected_patches(padded_img, ker, stride, dilation, return_i_j=False):
    h_ker, w_ker = ker
    h_stride, w_stride = stride
    h_dilation, w_dilation = dilation
    bz, img_c, img_h, img_w = padded_img.shape

    h_out = (img_h - h_dilation * (h_ker - 1) - 1) // h_stride + 1
    w_out = (img_w - w_dilation * (w_ker - 1) - 1) // w_stride + 1
    ker_hs = np.arange(h_ker) * h_dilation
    out_cnt = h_out * w_out
    all_hs = np.tile(np.repeat(ker_hs, w_ker), out_cnt)
    ker_ws = np.arange(w_ker) * w_dilation
    all_ws = np.tile(ker_ws, h_ker * out_cnt)
    h_shifts = np.repeat(np.arange(h_out), w_out * h_ker * w_ker) * h_stride
    w_shifts = np.repeat(np.tile(np.arange(w_out), h_out), h_ker * w_ker) * w_stride
    i = all_hs + h_shifts
    j = all_ws + w_shifts
    select_img = padded_img[:, :, i, j]
    res = select_img.reshape(bz, img_c, out_cnt, h_ker, w_ker)
    if return_i_j:
        return res, i, j
    return res


def check_selected_patches():
    dilation = (2, 1)
    stride = (1, 1)
    ker = (3, 3)
    bz, img_c, img_h, img_w = img_shape = (1, 2, 6, 6)
    img = np.arange(img_c * img_h * img_w).reshape(img_shape)
    print("Input image:")
    print(img)

    res = get_all_selected_patches(img, ker, stride, dilation)
    print("Result patches:")
    print(res)


if __name__ == "__main__":
    check_selected_patches()
