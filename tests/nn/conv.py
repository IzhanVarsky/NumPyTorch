import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestConv(unittest.TestCase):
    def test_selected_patches(self):
        dilation = (2, 1)
        stride = (1, 2)
        ker = (3, 4)
        img = np.array([[[[0, 1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10, 11],
                          [12, 13, 14, 15, 16, 17],
                          [18, 19, 20, 21, 22, 23],
                          [24, 25, 26, 27, 28, 29],
                          [30, 31, 32, 33, 34, 35]],

                         [[36, 37, 38, 39, 40, 41],
                          [42, 43, 44, 45, 46, 47],
                          [48, 49, 50, 51, 52, 53],
                          [54, 55, 56, 57, 58, 59],
                          [60, 61, 62, 63, 64, 65],
                          [66, 67, 68, 69, 70, 71]]]])
        res = nn.utils.get_all_selected_patches(img, ker, stride, dilation)

        ground_truth = np.array([[[[[0, 1, 2, 3],
                                    [12, 13, 14, 15],
                                    [24, 25, 26, 27]],

                                   [[2, 3, 4, 5],
                                    [14, 15, 16, 17],
                                    [26, 27, 28, 29]],

                                   [[6, 7, 8, 9],
                                    [18, 19, 20, 21],
                                    [30, 31, 32, 33]],

                                   [[8, 9, 10, 11],
                                    [20, 21, 22, 23],
                                    [32, 33, 34, 35]]],

                                  [[[36, 37, 38, 39],
                                    [48, 49, 50, 51],
                                    [60, 61, 62, 63]],

                                   [[38, 39, 40, 41],
                                    [50, 51, 52, 53],
                                    [62, 63, 64, 65]],

                                   [[42, 43, 44, 45],
                                    [54, 55, 56, 57],
                                    [66, 67, 68, 69]],

                                   [[44, 45, 46, 47],
                                    [56, 57, 58, 59],
                                    [68, 69, 70, 71]]]]])

        np.testing.assert_array_equal(res, ground_truth, err_msg="---- Test failed :(")
        print("++++ Test Passed!")

    def test_rand_conv_forward(self):
        ker = (3, 4)
        stride = (2, 5)
        padding = (3, 1)
        dilation = (4, 6)
        bz, in_channels, H, W = input_shape = (5, 3, 32, 32)
        input = np.random.random(input_shape) * 100
        out_channels = 7

        numpy_dtype = np.float32
        my_conv1 = nn.FastConv2d(in_channels, out_channels,
                                 kernel_size=ker, stride=stride, padding=padding,
                                 dilation=dilation, dtype=numpy_dtype)
        my_out = my_conv1(input)

        torch_dtype = torch.float32
        t_conv = torch.nn.Conv2d(in_channels, out_channels,
                                 kernel_size=ker, stride=stride, padding=padding,
                                 dilation=dilation, dtype=torch_dtype)
        t_conv.bias.data = torch.tensor(my_conv1.bias.data)
        t_conv.weight.data = torch.tensor(my_conv1.weight.data)
        t_out = t_conv(torch.tensor(input, dtype=torch_dtype))

        t_out_n = t_out.detach().numpy()
        assert_allclose(my_out, t_out_n, atol=0.0001, err_msg="---- Test failed :(")
        print("++++ Test Passed!")
