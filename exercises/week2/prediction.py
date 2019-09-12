import os
import numpy as np
from scipy import optimize


def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)

    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).

    X : array_like
        The data to use for computing predictions. The rows is the number
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X.

    Instructions
    ------------
    Complete the following code to make predictions using your learned
    logistic regression parameters.You should set p to a vector of 0's and 1's
    """
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    predictions = sigmoid(np.dot(X, theta))

    pos = predictions >= 0.5
    neg = predictions < 0.5

    for i in range(0, m):
        if pos[i]:
            p[i] = 1
        else:
            p[i] = 0

    return p


def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    g = 1 / (1 + np.e ** -z)

    return g


def costFunction(theta, X, y):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # a m x 1 matrix with our "linear" regression predictions
    # then apply our hypothesis (the sigmoid function) on all predictions
    predictions = sigmoid(np.dot(X, theta))

    class1_cost = -y * np.log(predictions)

    class0_cost = (1 - y) * np.log(1 - predictions)

    cost = class1_cost - class0_cost

    # cost = (-y * np.log(predictions) - (1 - y) * np.log(1 - predictions)).sum() / m -- as a one liner

    # cost is a m x 1 vector with each element being the actual cost for this specific example, so sum it up and divide
    # by number of training examples to get the average cost
    J = cost.sum() / m

    # a vectorized implementation of calculating the actual gradient
    grad = (np.dot(np.transpose(X), (predictions - y))) / m

    return J, grad


if __name__ == '__main__':
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise2/Data",
                                   "ex2data1.txt"), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]

    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to X
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # set options for optimize.minimize
    options = {'maxiter': 400}

    # see documention for scipy's optimize.minimize  for description about
    # the different parameters
    # The function returns an object `OptimizeResult`
    # We use truncated Newton algorithm for optimization which is
    # equivalent to MATLAB's fminunc
    # See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    res = optimize.minimize(costFunction,
                            initial_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of `OptimizeResult` object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property
    theta = res.x

    #  Predict probability for a student with score 45 on exam 1
    #  and score 85 on exam 2
    prob = sigmoid(np.dot([1, 45, 85], theta))
    print('For a student with scores 45 and 85,'
          'we predict an admission probability of {:.3f}'.format(prob))
    print('Expected value: 0.775 +/- 0.002\n')

    # Compute accuracy on our training set
    p = predict(theta, X)
    print('Train Accuracy: {:.2f} %'.format(np.mean(p == y) * 100))
    print('Expected accuracy (approx): 89.00 %')
