import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        m = target.shape[0]
        pred = softmax(x)
        log_likelihood = -np.log(pred[range(m), target])
        loss = np.sum(log_likelihood) / m

        grad = softmax(x)
        grad[range(m), target] -= 1
        grad = grad/m

        return (loss, grad)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = ((x - target)**2).mean()
        grad = 2*(x- target) / (x.shape[0]*x.shape[1])

        return (loss, grad)
