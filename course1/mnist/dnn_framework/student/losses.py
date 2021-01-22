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

        # pred = softmax(x)
        # y = np.zeros_like(x)
        # for i in range(len(target)):
        #     y[i][target[i]] = 1
        # loss = - np.sum(y * np.log(pred))

        # Loss
        m = target.shape[0]
        p = softmax(x)
        log_likelihood = -np.log(p[range(m), target])
        loss = np.sum(log_likelihood) / m

        # Gradient
        grad = softmax(x)
        grad[range(m), target] -= 1
        grad = grad / m

        return (loss, grad)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        exps = np.exp(x[i] - np.max(x[i]))
        y[i] = exps / np.sum(exps)
    return y


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
