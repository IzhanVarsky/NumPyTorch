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

        np.testing.assert_array_equal(res, ground_truth,
                                      err_msg="---- Selected patches test failed :(")
        print("++++ Selected patches test passed!")

    def test_rand_conv(self):
        ker = (3, 1)
        stride = (3, 2)
        padding = (2, 4)
        dilation = (1, 2)
        bz, in_channels, H, W = input_shape = (4, 3, 16, 16)
        input = np.random.random(input_shape)
        out_channels = 9

        numpy_dtype = np.float32
        torch_dtype = torch.float32
        my_layer = nn.Conv2d(in_channels, out_channels,
                             kernel_size=ker, stride=stride, padding=padding,
                             dilation=dilation, dtype=numpy_dtype)
        torch_layer = torch.nn.Conv2d(in_channels, out_channels,
                                      kernel_size=ker, stride=stride, padding=padding,
                                      dilation=dilation, dtype=torch_dtype)
        my_layer.bias.data = torch_layer.bias.data.numpy()
        my_layer.weight.data = torch_layer.weight.data.numpy()

        my_out = my_layer(input)
        grad = np.random.random((bz, out_channels, *my_out.shape[-2:]))
        my_grad_out = my_layer.backward(grad)

        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_layer.bias.grad, torch_layer.bias.grad.numpy(),
                        atol=0.001, err_msg="---- Backward: bias grad test failed :(")
        print("++++ Backward: bias grad test passed!")
        assert_allclose(my_layer.weight.grad.astype(numpy_dtype), torch_layer.weight.grad.numpy(),
                        atol=0.001, err_msg="---- Backward: weights grad test failed :(")
        print("++++ Backward: weights grad test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
