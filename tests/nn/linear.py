import random
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestLinear(unittest.TestCase):
    def test_rand_linear(self):
        bz = 4
        in_features = random.randint(5, 50)
        out_features = random.randint(5, 50)
        input = np.random.random((bz, in_features)) * 100

        grad = np.random.random((bz, out_features))

        numpy_dtype = np.float32
        my_linear = nn.Linear(in_features, out_features, dtype=numpy_dtype)
        my_out = my_linear(input)
        my_grad_out = my_linear.backward(grad)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_linear = torch.nn.Linear(in_features, out_features, dtype=torch_dtype)
        t_linear.bias.data = torch.tensor(my_linear.biases.value, requires_grad=True)
        t_linear.weight.data = torch.tensor(my_linear.weights.value, requires_grad=True)
        t_out = t_linear(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(t_linear.bias.grad.numpy(), my_linear.biases.grad,
                        atol=0.0001, err_msg="---- Backward: bias grad test failed :(")
        print("++++ Backward: bias grad test passed!")
        assert_allclose(t_linear.weight.grad.numpy(), my_linear.weights.grad,
                        atol=0.0001, err_msg="---- Backward: weights grad test failed :(")
        print("++++ Backward: weights grad test passed!")
        assert_allclose(t_input.grad.numpy(), my_grad_out,
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
