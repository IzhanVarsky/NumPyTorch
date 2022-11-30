import random
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestBatchNorm(unittest.TestCase):
    def test_rand_batchnorm(self):
        bz = 8
        in_features = random.randint(5, 50)
        input = np.random.random((bz, in_features)) * 100
        numpy_dtype = np.float32

        grad = np.random.random((bz, in_features))
        bias = np.random.random(in_features).astype(numpy_dtype)
        weight = np.random.random(in_features).astype(numpy_dtype)

        my_layer = nn.BatchNorm1d(in_features)
        my_layer.bias.data = bias[None, :]
        my_layer.weight.data = weight[None, :]
        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.BatchNorm1d(in_features)
        torch_layer.bias.data = torch.tensor(bias, dtype=torch_dtype, requires_grad=True)
        torch_layer.weight.data = torch.tensor(weight, dtype=torch_dtype, requires_grad=True)
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_layer.bias.grad[0], torch_layer.bias.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: bias grad test failed :(")
        print("++++ Backward: bias grad test passed!")
        assert_allclose(my_layer.weight.grad[0], torch_layer.weight.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: weights grad test failed :(")
        print("++++ Backward: weights grad test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
