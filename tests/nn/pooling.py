import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestPooling(unittest.TestCase):
    def test_rand_maxpool(self):
        ker = (5, 6)
        stride = (3, 2)
        padding = (2, 2)
        dilation = (2, 3)
        bz, in_channels, H, W = input_shape = (9, 6, 28, 24)
        input = np.random.random(input_shape) * 100

        torch_dtype = torch.float32
        my_layer = nn.MaxPool2d(kernel_size=ker, stride=stride, padding=padding,
                                dilation=dilation)
        torch_layer = torch.nn.MaxPool2d(kernel_size=ker, stride=stride, padding=padding,
                                         dilation=dilation)
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_out = torch_layer(t_input)
        grad = np.random.random((bz, in_channels, *t_out.shape[-2:]))
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.00001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.00001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
