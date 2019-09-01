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
    xTrans = X.transpose()
    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        # avg cost per example, just for debugging purposes
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient calculated as follows:
        # Remember that x0 = 1.
        # Summation over the loss * its corresponding features value (the partial derivation ath this point)
        # Summation by matrix multiplication: Need to transpose X for valid multiplication and
        # correct values.
        # Divide by m or multi with (1/m)
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient

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

def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """

    fig = pyplot.figure()

    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.xlabel("Population in 10000s")
    pyplot.ylabel("Revenue in 10000$")
    pyplot.show()


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

    X = np.stack([np.ones(97), X], axis=1)
    print("Normal Equation optimal values: {:.4f}, {:.4f}".format(*np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)))


    pyplot.plot(X[:, 1], y, 'ro', ms=10, mec='k')
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression']);
    pyplot.show()


