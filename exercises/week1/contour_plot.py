# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces



def contour():
    # grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

    # Fill out J_vals
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = computeCost(X, y, [theta0, theta1])

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals.T

    # surface plot
    fig = pyplot.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.title('Surface')

    # contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    ax = pyplot.subplot(122)
    pyplot.contour(theta0_vals, theta1_vals, J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
    pyplot.xlabel('theta0')
    pyplot.ylabel('theta1')
    pyplot.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
    pyplot.title('Contour, showing minimum')