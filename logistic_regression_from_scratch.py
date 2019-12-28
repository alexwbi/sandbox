import numpy as np

DEFAULT_COVARIANCE = [[1, 2], [2, 1]]


def predict(x, weights, intercept):
    p_success = np.array([_sigmoid(x.dot(weights) + intercept)])
    return np.where(p_success > 0.5, 1, 0)


def _generate_data(random_seed=10):
    np.random.seed(random_seed)
    a = np.random.multivariate_normal(mean=[0, 0], cov=DEFAULT_COVARIANCE, size=N)
    b = np.random.multivariate_normal(mean=[2, 2], cov=DEFAULT_COVARIANCE, size=N)
    x = np.vstack((a, b))
    y = np.hstack((np.zeros(N), np.ones(N)))
    return x, y


def _derivative(x, y, p):
    """ Given input x, output y, and probability p, calculate first derivative of likelihood function """
    return sum((y - p) * x)


def _cross_entropy_loss(y_pred, y_actual):
    return -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
