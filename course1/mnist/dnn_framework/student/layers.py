

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
        """
        This method initializes the parameters of the layer.
        """
        self.params = {}

        # Initialize weights with a Xavier distribution
        self.params['w'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / (self.input_size + self.output_size)),
                          size=(self.output_size, self.input_size))
        self.params['b'] = np.random.normal(loc=0.0,
                          scale=np.sqrt(2 / self.output_size),
                          size=(self.output_size,))


    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return self.params


    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """        
        y = np.matmul(x, self.params['w'].T) + self.params['b']
        return (y, x)


    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key.
        """
        x_grad = np.matmul(output_grad, self.params['w'])
        w_grad = np.matmul(output_grad.T, cache)
        b_grad = np.sum(output_grad, axis=0)
        return (x_grad, {'w': w_grad, 'b': b_grad})


    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}



class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """
    def __init__(self, input_size, alpha=0.1):
        super().__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.init_params()
        self.init_buffers()


    def init_params(self):
        """
        This method initializes the parameters of the layer.
        """
        self.params = {}
        self.params['gamma'] = np.ones(self.input_size)
        self.params['beta'] = np.zeros(self.input_size)


    def init_buffers(self):
        """
        This method initializes the buffers of the layer.
        """
        self.buffers = {'global_mean': np.zeros(self.input_size), 
                        'global_variance': np.zeros(self.input_size)}


    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return self.params


    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return self.buffers


    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)


    def _forward_training(self, x):
        """
        This method performs the forward pass of the layer when in training mode.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        # Calculate mean and variance of batch
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Update buffers
        self.buffers['global_mean'] = (1 - self.alpha)*self.buffers['global_mean'] \
                                          + self.alpha*mean
        self.buffers['global_variance'] = (1 - self.alpha)*self.buffers['global_variance'] \
                                          + self.alpha*var

        # Normalize batch
        eps = 7./3 - 4./3 -1
        x_hat = (x - mean) / (np.sqrt(var + eps))

        # Adjust value with layer weights
        y = self.params['gamma']*x_hat + self.params['beta']
        return (y, [x, x_hat, mean, var, eps])


    def _forward_evaluation(self, x):
        """
        This method performs the forward pass of the layer when in evaluation mode.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        # Fetch global mean and variance values
        eps = 7./3 - 4./3 -1
        mean = self.buffers['global_mean']
        var = self.buffers['global_variance']

        # Normalize input        
        x_hat = (x - mean) / (np.sqrt(var + eps))

        # Adjust value with layer weights
        y = self.params['gamma']*x_hat + self.params['beta']
        return (y, [x, x_hat, mean, var, eps])


    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key.
        """
        # Unpack cache 
        x, x_hat, mean, var, eps = cache
        gamma = self.params['gamma']
        beta = self.params['beta']

        # Gamma
        g_grad = (output_grad*x_hat).sum(axis=0) #TODO
        
        # Beta
        b_grad = output_grad.sum(axis=0)

        # X (Simplified formula: see 
        # https://zaffnet.github.io/batch-normalization?fbclid=IwAR3-2dabx7MvD_kGB_oEmDQEtvksqRplDyqCfe0OJ0p1R0SbDFMws8Ndxvc#bprop)
        m = x.shape[0]
        t = 1./np.sqrt(var + eps)

        x_grad = (gamma * t / m) * (m * output_grad - np.sum(output_grad, axis=0)
             - t**2 * (x - mean) * np.sum(output_grad*(x - mean), axis=0))
        
        return x_grad, {'gamma': g_grad, 'beta': b_grad}



class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """
    def __init__(self):
        super().__init__()  

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        y = 1/(1 + np.exp(-x))
        return (y, y)

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key.
        """
        dydx = (1 - cache)*cache
        grad = output_grad*dydx
        return (grad, {})

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        y = x.copy()
        y[y < 0] = 0
        return (y, x) 

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key.
        """
        grad = output_grad.copy()
        grad[cache < 0] = 0
        return (grad, {}) 

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        return {}

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        return {}
