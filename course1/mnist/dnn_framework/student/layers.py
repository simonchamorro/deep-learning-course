

import numpy as np
from dnn_framework import Layer



class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_params()

    def init_params(self):
        self.params = {}
        self.params['w'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (self.input_size + self.output_size)),
                          size=(self.input_size, self.output_size))
        self.params['b'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / self.output_size),
                          size=(self.output_size,))

    def get_parameters(self):
        return self.params

    def forward(self, x):
        y = np.matmul(x, self.params['w']) + self.params['b']
        return (y, x)

    def backward(self, output_grad, cache):
        dx = np.matmul(output_grad, np.transpose(self.params['w']))
        dw = np.matmul(np.transpose(cache), output_grad)
        db = np.sum(output_grad, axis=0)
        return (dx, {'w': dw, 'b': db})


class BatchNormalization(Layer):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.init_params()

    def init_params(self):
        self.params = {}
        self.params['gamma'] = np.ones(self.input_size)
        self.params['beta'] = np.zeros(self.input_size)

    def get_parameters(self):
        return self.params

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key.
        """
        raise NotImplementedError()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()  

    def forward(self, x):
        y = 1/(1 + np.exp(-x))
        return (y, y)

    def backward(self, output_grad, cache):
        dydx = (1 - cache)*cache
        grad = output_grad*dydx
        return (grad, {})


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = x.copy()
        y[y < 0] = 0
        return (y, x) 

    def backward(self, output_grad, cache):
        grad = output_grad.copy()
        grad[cache < 0] = 0
        return (grad, {}) 