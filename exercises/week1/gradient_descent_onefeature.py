# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : arra_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()

    X = np.stack([np.ones(m), X], axis=1)

    J_history = []  # Use a python list to save cost in every iteration

    for i in range(num_iters):

        predictions = np.dot(X, theta[:, None])
        theta[0] = theta[0] - alpha * (1/m) * (np.sum(np.subtract(predictions, y)))

        temp_theta_one = 0.0

        for j in range(0, m):
            temp_theta_one += ((predictions[j][0] - y[j]) * X[j][1])
        theta[1] = theta[1] - (temp_theta_one * alpha * (1/m))

        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history


def computeCost(X, y, theta):

    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # Add a column of ones to X. The numpy function stack joins arrays along a given axis.
    # The first axis (axis=0) refers to rows (training examples)
    # and second axis (axis=1) refers to columns (features).

    for i in range(0, m):
        J += ((theta[0] + theta[1] * X[i][1]) - y[i]) ** 2
        # for more than one feature I´d simply implement it with a second for loop

    return J * (1/(2 * m))


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise1/Data",
                                   "ex1data1.txt"), delimiter=',')

    data_test = [[], []]
    X, y = data[:, 0], data[:, 1]

    # initialize fitting parameters
    theta = np.zeros(2)

    # some gradient descent settings
    iterations = 1500
    alpha = 0.01

    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
    print('Expected theta values (approximately): [-3.6303, 1.1664]')

