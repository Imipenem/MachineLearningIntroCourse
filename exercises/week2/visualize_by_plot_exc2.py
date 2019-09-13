import os
import numpy as np
from matplotlib import pyplot


def plotData(X, y):

    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], 'og')
    pyplot.plot(X[neg, 0], X[neg, 1], 'ro')
    pyplot.xlabel("First test")
    pyplot.ylabel("Second test")
    pyplot.show()


if __name__ == '__main__':
    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise2/Data",
                                   "ex2data2.txt"), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    plotData(X,y)