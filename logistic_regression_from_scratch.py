import numpy as np

DEFAULT_COVARIANCE = [[1, 2], [2, 1]]


def _generate_data(random_seed=10):
    np.random.seed(random_seed)
    a = np.random.multivariate_normal(mean=[0, 0], cov=DEFAULT_COVARIANCE, size=N)
    b = np.random.multivariate_normal(mean=[2, 2], cov=DEFAULT_COVARIANCE, size=N)
    x = np.vstack((a, b))
    y = np.hstack((np.zeros(N), np.ones(N)))
    return x, y


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
