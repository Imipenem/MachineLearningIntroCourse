import os
import numpy as np
from matplotlib import pyplot


def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.

    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total
        number of polynomial features.

    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).

    y : array_like
        The data labels. A vector with shape (m, ).

    lambda_ : float
        The regularization parameter.

    Returns
    -------
    J : float
        The computed value for the regularized cost function.

    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.

    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    predictions = sigmoid(np.dot(X, theta))

    class1_cost = -y * np.log(predictions)

    class0_cost = (1 - y) * np.log(1 - predictions)

    cost = class1_cost - class0_cost

    # cost = (-y * np.log(predictions) - (1 - y) * np.log(1 - predictions)).sum() / m -- as a one liner

    # cost is a m x 1 vector with each element being the actual cost for this specific example, so sum it up and divide
    # by number of training examples to get the average cost
    J = cost.sum() / m

    # take each regularized element into account BUT theta[0]
    reg = ((theta[1:] ** 2).sum()) * (lambda_/(2*m))
    J += reg

    # first, calculate the gradient as usual for logistic regression
    grad = (np.dot(np.transpose(X), (predictions - y))) / m

    # regularize each element BUT NOT theta[0] the first element
    grad[1:] = grad[1:] + (lambda_ / m) * theta[1:]

    return J, grad


def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    g = 1 / (1 + np.e ** -z)

    return g


def mapFeature(X1, X2, degree=6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


if __name__ == '__main__':
    # Load Data
    # The first two columns contains the X values and the third column
    # contains the label (y).
    data = np.loadtxt(os.path.join("/home/thelichking/Desktop/ml-coursera-python-assignments/Exercise2/Data",
                                   "ex2data2.txt"), delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    X = mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

    print('Cost at initial theta (zeros): {:.3f}'.format(cost))
    print('Expected cost (approx)       : 0.693\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')

    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones(X.shape[1])
    cost, grad = costFunctionReg(test_theta, X, y, 10)

    print('------------------------------\n')
    print('Cost at test theta    : {:.2f}'.format(cost))
    print('Expected cost (approx): 3.16\n')

    print('Gradient at initial theta (zeros) - first five values only:')
    print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
    print('Expected gradients (approx) - first five values only:')
    print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')