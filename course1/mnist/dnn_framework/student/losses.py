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

        # Convert label to one-hot 
        y = np.zeros_like(x)
        for i in range(len(target)):
            y[i][target[i]] = 1

        # Loss

        # loss = - np.sum(y * np.log(pred))

        # Simplified version
        # https://deepnotes.io/softmax-crossentropy?fbclid=IwAR1P3EhtYpPfri0cS4qNhWuBTovGYctWmvyx8HofCXjCWuxSF9yNVkAT6wU
        log_likelihood = -np.log(pred[range(m), target])
        loss = np.sum(log_likelihood)


        # Gradient

        # # Calculate cross entropy grad
        # ce_grad = -y/pred
        # grad = np.zeros_like(ce_grad)

        # # For each input, calculate d matrix and softmax grad
        # for n in range(m):
        #     d_matrix = np.zeros((x.shape[1], x.shape[1]))
            
        #     for i in range(x.shape[1]):
        #         for j in range(x.shape[1]):
        #             if i == j:
        #                 d_matrix[i,j] = pred[n,j]*(1-pred[n,j])
        #             else:
        #                 d_matrix[i,j] = -pred[n,j]*pred[n,i]

        #     grad[n] = np.matmul(d_matrix, ce_grad[n])

        # Simplified version
        # https://deepnotes.io/softmax-crossentropy?fbclid=IwAR1P3EhtYpPfri0cS4qNhWuBTovGYctWmvyx8HofCXjCWuxSF9yNVkAT6wU
        grad = pred    
        grad[range(m), target] -= 1

        return (loss, grad)


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    y = np.zeros_like(x)

    # Calculate softmax for each input vector
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
