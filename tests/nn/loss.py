import random
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestLosses(unittest.TestCase):
    def test_rand_cross_entropy_loss(self):
        softmax = nn.Softmax()
        in_shape = (12, 10)
        input = np.random.random(in_shape)
        target = softmax(np.random.random(in_shape))

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_target = torch.tensor(target, dtype=torch_dtype)
        torch_layer = torch.nn.CrossEntropyLoss()
        t_out = torch_layer(t_input, t_target)
        t_out.backward(retain_graph=True)

        my_layer = nn.Net(nn.Softmax(), nn.CrossEntropyLoss())
        my_out = my_layer(input)
        my_out = my_layer.loss(my_out, target)
        my_grad_out = my_layer.backward()

        assert_allclose(my_out.mean(), t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")

    def test_rand_mse_loss(self):
        in_shape = (3, 7, 6)
        input = np.random.random(in_shape)
        target = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        t_target = torch.tensor(target, dtype=torch_dtype)
        torch_layer = torch.nn.MSELoss()
        t_out = torch_layer(t_input, t_target)
        t_out.backward(retain_graph=True)

        my_layer = nn.MSELoss()
        my_out = my_layer(input, target)
        my_grad_out = my_layer.backward()

        assert_allclose(my_out.mean(), t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
