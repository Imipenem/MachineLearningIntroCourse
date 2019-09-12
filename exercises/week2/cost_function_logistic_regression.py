import numpy as np
import os


def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    g = 1 / (1 + np.e ** -z)

    return g


def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression.

    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).

    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the
        intercept has already been added to the input.

    y : arra_like
        Labels for the input. This is a vector of shape (m, ).

    Returns
    -------
    J : float
        The computed value for the cost function.

    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
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

    cost, grad = costFunction(initial_theta, X, y)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx): 0.693\n')

    print('Gradient at initial theta (zeros):')
    print('\t[{:.4f}, {:.4f}, {:.4f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[-0.1000, -12.0092, -11.2628]\n')

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost, grad = costFunction(test_theta, X, y)

    print('Cost at test theta: {:.3f}'.format(cost))
    print('Expected cost (approx): 0.218\n')

    print('Gradient at test theta:')
    print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*grad))
    print('Expected gradients (approx):\n\t[0.043, 2.566, 2.647]')
