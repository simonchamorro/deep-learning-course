import unittest

import os
import sys
import numpy as np
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/dnn_framework')

from dnn_framework import FullyConnectedLayer, BatchNormalization, Sigmoid, ReLU
from tests import test_layer_input_grad, test_layer_parameter_grad


class LayerTestCase(unittest.TestCase):
    def test_fully_connected_layer_forward(self):
        layer = FullyConnectedLayer(2, 1)
        layer.get_parameters()['w'][:] = np.array([[2], [3]])
        layer.get_parameters()['b'][:] = np.array([1])
        x = np.array([[-1.0, 0.5]])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(y[0], 0.5)

    def test_fully_connected_layer_forward_backward(self):
        self.assertTrue(test_layer_input_grad(FullyConnectedLayer(4, 10), (2, 4)))
        self.assertTrue(test_layer_parameter_grad(FullyConnectedLayer(4, 10), (2, 4), 'w'))
        self.assertTrue(test_layer_parameter_grad(FullyConnectedLayer(4, 10), (2, 4), 'b'))

    def test_batch_normalization_forward_training(self):
        layer = BatchNormalization(2)
        layer.get_parameters()['gamma'][:] = np.array([1, 2])
        layer.get_parameters()['beta'][:] = np.array([-1, 1])

        x = np.array([[-1, -2], [1, -1], [0, -1.5]])
        y, _ = layer.forward(x)

        expected_y = np.array([[-2.22474487, -1.44948974], [0.22474487, 3.44948974], [-1.0, 1.0]])
        self.assertAlmostEqual(np.mean(y - expected_y), 0.0)

    def test_batch_normalization_forward_evaluation(self):
        layer = BatchNormalization(2)
        layer.eval()
        layer.get_buffers()['global_mean'][:] = np.array([0.0, -1.5])
        layer.get_buffers()['global_variance'][:] = np.array([0.81649658, 0.40824829])
        layer.get_parameters()['gamma'][:] = np.array([1, 2])
        layer.get_parameters()['beta'][:] = np.array([-1, 1])

        x = np.array([[-1, -2], [1, -1], [0, -1.5]])
        y, _ = layer.forward(x)

        expected_y = np.array([[-2.22474487, -1.44948974], [0.22474487, 3.44948974], [-1.0, 1.0]])
        self.assertAlmostEqual(np.mean(y - expected_y), 0.0)

    def test_batch_normalization_backward(self):
        self.assertTrue(test_layer_input_grad(BatchNormalization(4), (2, 4)))
        self.assertTrue(test_layer_parameter_grad(BatchNormalization(4), (2, 4), 'gamma'))
        self.assertTrue(test_layer_parameter_grad(BatchNormalization(4), (2, 4), 'beta'))

    def test_sigmoid_forward(self):
        layer = Sigmoid()
        x = np.array([-1.0, 0.5])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(y[0], 0.2689414)
        self.assertAlmostEqual(y[1], 0.6224593)

    def test_sigmoid_backward(self):
        self.assertTrue(test_layer_input_grad(Sigmoid(), (2, 3)))

    def test_relu_forward(self):
        layer = ReLU()
        x = np.array([-1.0, 0.5])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(x[0], -1.0)
        self.assertAlmostEqual(x[1], 0.5)
        self.assertAlmostEqual(y[0], 0.0)
        self.assertAlmostEqual(y[1], 0.5)

    def test_relu_backward(self):
        self.assertTrue(test_layer_input_grad(ReLU(), (2, 3)))


if __name__ == '__main__':
    unittest.main()
