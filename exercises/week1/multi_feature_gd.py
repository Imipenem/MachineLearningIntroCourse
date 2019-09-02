# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        A vector of shape (m, ) for the values at a given data point.

    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )

    alpha : float
        The learning rate for gradient descent.

    num_iters : int
        The number of iterations to run gradient descent.

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

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()

    J_history = []
    xTrans = X.transpose()
    for i in range(num_iters):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
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
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

def plot_learning_rates():
    """
    Instructions
    ------------
    We have provided you with the following starter code that runs
    gradient descent with a particular learning rate (alpha).

    Your task is to first make sure that your functions - `computeCost`
    and `gradientDescent` already work with  this starter code and
    support multiple variables.

    After that, try running gradient descent with different values of
    alpha and see which one gives you the best result.

    Finally, you should complete the code at the end to predict the price
    of a 1650 sq-ft, 3 br house.

    Hint
    ----
    At prediction, make sure you do the same feature normalization.
    """
    # Choose some alpha value - change this
    alpha = 0.
    num_iters = 100

    # init theta and run gradient descent
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')

    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(theta)))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ======================= YOUR CODE HERE ===========================
    # Recall that the first column of X is all-ones.
    # Thus, it does not need to be normalized.

    price = 0  # You should change this

    # ===================================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))

def computeCostMulti(X, y, theta):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    return (np.dot((np.transpose(np.dot(X, theta)) - y), (np.dot(X, theta) - y))) / (2 * m)


def featureNormalize(X):

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
    X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

    alpha = 0.
    num_iters = 100

    # init theta and run gradient descent
    theta = np.zeros(4)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
    pyplot.xlabel('Number of iterations')
    pyplot.ylabel('Cost J')
    pyplot.show()

    # Display the gradient descent's result
    print('theta computed from gradient descent: {:s}'.format(str(theta)))