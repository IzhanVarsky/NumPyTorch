import random
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestActivations(unittest.TestCase):
    def test_rand_relu(self):
        in_shape = (3, 6)
        input = (np.random.random(in_shape) - 0.5) * 10
        grad = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.ReLU()
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_layer = nn.ReLU()
        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")

    def test_rand_sigmoid(self):
        in_shape = (3, 6)
        input = (np.random.random(in_shape) - 0.5) * 10
        grad = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.Sigmoid()
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_layer = nn.Sigmoid()
        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")

    def test_rand_tanh(self):
        in_shape = (3, 6)
        input = (np.random.random(in_shape) - 0.5) * 10
        grad = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.Tanh()
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_layer = nn.Tanh()
        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")

    def test_rand_softmax(self):
        in_shape = (3, 5)
        input = (np.random.random(in_shape) - 0.5) * 10
        grad = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.Softmax()
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_layer = nn.Softmax()
        my_out = my_layer(input)
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_out, t_out.detach().numpy(), atol=0.0001,
                        err_msg="---- Forward: test failed :(")
        print("++++ Forward: test passed!")
        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
