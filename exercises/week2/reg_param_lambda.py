import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize


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


def plotData(X, y):

    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], 'og')
    pyplot.plot(X[neg, 0], X[neg, 1], 'ro')
    pyplot.xlabel("First test")
    pyplot.ylabel("Second test")
    pyplot.show()


def plotDecisionBoundary(theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.

    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.

    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).

    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.

    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        pyplot.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        pyplot.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        pyplot.xlim([30, 100])
        pyplot.ylim([30, 100])


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

    # Set regularization parameter lambda to 1 (you should vary this)
    lambda_ = 1

    # set options for optimize.minimize
    options = {'maxiter': 100}

    res = optimize.minimize(costFunctionReg,
                            initial_theta,
                            (X, y, lambda_),
                            jac=True,
                            method='TNC',
                            options=options)

    # the fun property of OptimizeResult object returns
    # the value of costFunction at optimized theta
    cost = res.fun

    # the optimized theta is in the x property of the result
    theta = res.x

    plotDecisionBoundary(theta, X, y)
    pyplot.xlabel('Microchip Test 1')
    pyplot.ylabel('Microchip Test 2')
    pyplot.legend(['y = 1', 'y = 0'])
    pyplot.grid(False)
    pyplot.title('lambda = %0.2f' % lambda_)

    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
    print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')
