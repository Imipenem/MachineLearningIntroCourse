# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X_norm, axis=0)
    X_norm = X_norm - mu

    sigma = np.std(X, axis=0)

    X_norm = X_norm/sigma

    return X_norm, mu, sigma


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise1/Data",
                                   "ex1data2.txt"), delimiter=',')
    X = data[:, :3]  # the first two rows
    y = data[:, 2]  # the actual 3rd row
    m = y.size

    X_norm, mu, sigma = featureNormalize(X)
    print(mu, sigma)



