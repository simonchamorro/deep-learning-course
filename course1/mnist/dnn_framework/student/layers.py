

import numpy as np
from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init_params()


    def init_params(self):
        self.params = {}
        self.params['w'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (self.input_size + self.output_size)),
                          size=(self.output_size, self.input_size))
        self.params['b'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / self.output_size),
                          size=(self.output_size,))


    def get_parameters(self):
        return self.params


    def forward(self, x):
        y = np.matmul(x, self.params['w'].T) + self.params['b']
        return (y, x)


    def backward(self, output_grad, cache):
        x_grad = np.matmul(output_grad, self.params['w'])
        w_grad = np.matmul(output_grad.T, cache)
        b_grad = np.sum(output_grad, axis=0)
        return (x_grad, {'w': w_grad, 'b': b_grad})


    def get_buffers(self):
        return {}



class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """
    def __init__(self, input_size, alpha=0.1):
        super().__init__()
        self.input_size = input_size
        self.init_params(alpha)
        self.init_buffers()


    def init_params(self, alpha):
        self.params = {}
        self.params['alpha'] = alpha
        self.params['gamma'] = np.ones(self.input_size)
        self.params['beta'] = np.zeros(self.input_size)


    def init_buffers(self):
        self.buffers = {'global_mean': np.zeros(self.input_size), 
                        'global_variance': np.zeros(self.input_size)}


    def get_parameters(self):
        return self.params


    def get_buffers(self):
        return self.buffers


    def forward(self, x):
        if self.is_training:
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)


    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Update buffers
        self.buffers['global_mean'] = (1 - self.params['alpha'])*self.buffers['global_mean'] \
                                          + self.params['alpha']*mean
        self.buffers['global_variance'] = (1 - self.params['alpha'])*self.buffers['global_variance'] \
                                          + self.params['alpha']*var

        # Normalize batch
        eps = 7./3 - 4./3 -1
        y = (x - mean) / (np.sqrt(var + eps))
        y = self.params['gamma']*y + self.params['beta']
        return (y, y)


    def _forward_evaluation(self, x):
        eps = 7./3 - 4./3 -1
        mean = self.buffers['global_mean']
        var = self.buffers['global_variance']
        y = (x - mean) / (np.sqrt(var + eps))
        y = self.params['gamma']*y + self.params['beta']
        return (y, y)


    def backward(self, output_grad, cache):
        x_hat_grad = output_grad*self.params['gamma']
        x_grad = 0 #TODO
        g_grad = (output_grad*x_hat_grad).sum(axis=0) #TODO
        b_grad = output_grad.sum(axis=0)
        return x_grad, {'gamma': g_grad, 'beta': b_grad}



class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """
    def __init__(self):
        super().__init__()  

    def forward(self, x):
        y = 1/(1 + np.exp(-x))
        return (y, y)

    def backward(self, output_grad, cache):
        dydx = (1 - cache)*cache
        grad = output_grad*dydx
        return (grad, {})

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """
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

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}
