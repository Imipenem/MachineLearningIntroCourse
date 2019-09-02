# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np


def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    Returns
    -------
    J : float
        The value of the cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variable correctly
    J = 0

    J = (np.dot((np.transpose(np.dot(X, theta)) - y), (np.dot(X, theta) - y))) / (2 * m)

    return J


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise1/Data",
                                   "ex1data2.txt"), delimiter=',')
    X = data[:, :3]  # the first two rows
    y = data[:, 2]  # the actual 3rd row
    m = y.size

    print(computeCostMulti(X,y,theta=np.array([-1, 2, 3])))
