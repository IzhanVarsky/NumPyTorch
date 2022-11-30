import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestPooling(unittest.TestCase):
    def test_rand_maxpool(self):
        ker = (3, 3)
        stride = (3, 2)
        padding = (1, 0)
        dilation = (1, 2)
        bz, in_channels, H, W = input_shape = (4, 3, 16, 16)
        input = np.random.random(input_shape)

        torch_dtype = torch.float32
        my_layer = nn.MaxPool2d(kernel_size=ker, stride=stride, padding=padding,
                                dilation=dilation)
        torch_layer = torch.nn.MaxPool2d(kernel_size=ker, stride=stride, padding=padding,
                                         dilation=dilation)

        my_out = my_layer(input)
        grad = np.random.random((bz, in_channels, *my_out.shape[-2:]))
        my_grad_out = my_layer.backward(grad)

        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
