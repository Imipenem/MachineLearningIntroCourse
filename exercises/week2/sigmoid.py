import numpy as np
import os


def sigmoid(z):
    """
    Compute sigmoid function given the input z.

    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector
        or a 2-D matrix.

    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.

    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    g = 1 / (1 + np.e ** -z)

    return g


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise2/Data",
                                   "ex2data1.txt"), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]

    print(sigmoid(0))
    print(sigmoid(1918398131881))
    print(sigmoid(np.array([[1,2,3,4,5],[2,3,4,5,6]])))
