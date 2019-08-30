# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


def computeCost(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
    """

    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # Add a column of ones to X. The numpy function stack joins arrays along a given axis.
    # The first axis (axis=0) refers to rows (training examples)
    # and second axis (axis=1) refers to columns (features).
    X = np.stack([np.ones(m), X], axis=1)

    for i in range(0, m):
        J += ((theta[0] + theta[1] * X[i][1]) - y[i]) ** 2
        # for more than one feature I´d simply implement it with a second for loop

    return J * (1/(2 * m))


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise1/Data",
                                   "ex1data1.txt"), delimiter=',')

    data_test = [[], []]
    X, y = data[:, 0], data[:, 1]

    J = computeCost(X, y, theta=np.array([0.0, 0.0]))
    print('With theta = [0, 0] \nCost computed = %.2f' % J)
    print('Expected cost value (approximately) 32.07\n')

    # further testing of the cost function
    J = computeCost(X, y, theta=np.array([-1, 2]))
    print('With theta = [-1, 2]\nCost computed = %.2f' % J)
    print('Expected cost value (approximately) 54.24')