import os
from matplotlib import pyplot
import numpy as np


def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data
    points with * for the positive examples and o for the negative examples.

    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset.

    y : array_like
        Label values for the dataset. A vector of size (M, ).

    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.
    """
    # Create New Figure
    fig = pyplot.figure()

    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], 'og')
    pyplot.plot(X[neg, 0], X[neg, 1], 'ro')
    pyplot.xlabel("First Exam")
    pyplot.ylabel("Second Exam")
    pyplot.show()


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise2/Data",
                                   "ex2data1.txt"), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]  # X as: Select Col 0 to 2 (exclusive 2) y as select col 2
    plotData(X,y)
