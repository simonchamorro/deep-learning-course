import unittest

import numpy as np

from dnn_framework import SgdOptimizer


class SgdOptimizerTestCase(unittest.TestCase):
    def test_step(self):
        parameters = {
            'x': np.array([10.0]),
            'y': np.array([10.0])
        }
        optimizer = SgdOptimizer(parameters, learning_rate=0.1)

        for i in range(100):
            parameter_grads = {
                'x': 2 * parameters['x'] + 1,  # f(x) = x^2 + x
                'y': 2 * parameters['y']  # f(y) = y^2
            }
            optimizer.step(parameter_grads)

        self.assertAlmostEqual(parameters['x'][0], -0.5)
        self.assertAlmostEqual(parameters['y'][0], 0.0)


if __name__ == '__main__':
    unittest.main()
