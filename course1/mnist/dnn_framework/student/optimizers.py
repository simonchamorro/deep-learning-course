from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        self._parameters = parameters
        self.learning_rate = learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        return parameter - parameter_grad*self.learning_rate
