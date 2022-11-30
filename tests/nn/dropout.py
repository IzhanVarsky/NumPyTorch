import random
import unittest

import numpy as np
import torch
from numpy.testing import assert_allclose

import nn


class TestDropout(unittest.TestCase):
    def test_rand_dropout_backward(self):
        in_shape = (8, 10, 12)
        p = random.random()
        input = np.random.random(in_shape) * 100
        grad = np.random.random(in_shape)

        torch_dtype = torch.float32
        t_input = torch.tensor(input, dtype=torch_dtype, requires_grad=True)
        torch_layer = torch.nn.Dropout(p)
        t_out = torch_layer(t_input)
        t_out.backward(torch.tensor(grad), retain_graph=True)

        my_layer = nn.Dropout(p)
        my_layer.cached_mask = t_out.detach().numpy() != 0
        my_grad_out = my_layer.backward(grad)

        assert_allclose(my_grad_out, t_input.grad.numpy(),
                        atol=0.0001, err_msg="---- Backward: input grad test failed :(")
        print("++++ Backward: input grad test passed!")
