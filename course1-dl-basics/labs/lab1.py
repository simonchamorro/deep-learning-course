import math
import numpy as np

#!/usr/bin/env python 

# Created on January 19 2021
# @author: Simon Chamorro       simon.chamorro@usherbrooke.ca

"""
------------------------------------
Lab 1 of class Machine Learning I

"""

import math
import numpy as np
import matplotlib.pyplot as plt 


def inv_matrix_gd(matrix, iterations=1000, lr=0.005):
    """
    Inverse a square matrix by using Gradient Descent
    """

    # Init inverse matrix with zeros
    loss = []
    inv = np.zeros(matrix.shape)
    identity = np.identity(matrix.shape[0])

    # Optimize inverse matrix
    for i in range(iterations):

        # Calculate loss and gradient
        l_matrix = (np.matmul(inv, matrix) - identity)**2
        l = l_matrix.sum()
        gradient = np.matmul( 2*(np.matmul(inv, matrix) - identity),
                              np.transpose(matrix))

        # Compute new inverse matrix
        inv = inv - lr*gradient
        loss.append(l)

    return inv, loss


def main():

    # Set print options
    np.set_printoptions(suppress=True)
    
    # Q1: Inverse matrix 1
    m1 = np.array([[3, 4, 1],
                   [5, 2, 3],
                   [6, 2, 2]])

    m1_inv = np.array([[-1/12, -1/4,  5/12],
                       [  1/3,    0,  -1/6],
                       [-1/12,  3/4, -7/12]])


    # Question 1 answers
    print('-------------------')
    print('Q1: Answer')
    print(m1_inv)
    print('Q1: 0.005')
    m1_inv_gd, loss = inv_matrix_gd(m1, lr=0.005)
    print(np.isclose(m1_inv, m1_inv_gd, atol=1e-03))
    plt.figure('Q1: lr 0.005')
    plt.plot(loss)

    print('Q1: 0.001')
    m1_inv_gd, loss = inv_matrix_gd(m1, lr=0.001)
    print(np.isclose(m1_inv, m1_inv_gd, atol=1e-03))
    plt.figure('Q1: lr 0.001')
    plt.plot(loss)

    print('Q1: 0.01')
    m1_inv_gd, loss = inv_matrix_gd(m1, lr=0.01)
    print(np.isclose(m1_inv, m1_inv_gd, atol=1e-03))
    plt.figure('Q1: lr 0.01')
    plt.plot(loss)

    # Q2: Inverse matrix 2
    print('-------------------')
    print('Q2')
    m2 = np.array([[3, 4, 1, 2, 1, 5],
                   [5, 2, 3, 2, 2, 1],
                   [6, 2, 2, 6, 4, 5],
                   [1, 2, 1, 3, 1, 2],
                   [1, 5, 2, 3, 3, 3],
                   [1, 2, 2, 4, 2, 1]])
    m2_inv_gd, loss = inv_matrix_gd(m2, lr=0.0025, iterations=50000)
    plt.figure('Q2: lr 0.0025')
    plt.plot(loss)
    plt.figure('Q2: Matrix')
    plt.imshow(np.matmul(m2, m2_inv_gd))

    # Q3: Inverse matrix 3
    print('-------------------')
    print('Q3')
    m3 = np.array([[2, 1, 1, 2],
                   [1, 2, 3, 2],
                   [2, 1, 1, 2],
                   [3, 1, 4, 1]])
    m3_inv_gd, loss = inv_matrix_gd(m3, lr=0.0025, iterations=5000)
    plt.figure('Q3: lr 0.0025')
    plt.plot(loss)
    plt.figure('Q3: Matrix')
    plt.imshow(np.matmul(m3, m3_inv_gd))

    plt.show()



if __name__ == '__main__':
    main()