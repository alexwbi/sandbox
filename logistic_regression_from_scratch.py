import numpy as np

DEFAULT_COVARIANCE = [[1, 2], [2, 1]]


def fit(X, y, random_seed, include_intercept=True, num_iterations=100000, min_threshold=1e-4):
    if include_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    initial_weights = np.random.random((X.shape[1], 1))
    initial_bias = 0.5

    initial_weights = np.random
    intercept =
    for i in range(num_iterations):

    pass


def predict(x, weights, intercept):
    p_success = np.array([_sigmoid(x.dot(weights) + intercept)])
    return np.where(p_success > 0.5, 1, 0)


def _generate_data(n, random_seed):
    np.random.seed(random_seed)
    a = np.random.multivariate_normal(mean=[0, 0], cov=DEFAULT_COVARIANCE, size=n)
    b = np.random.multivariate_normal(mean=[2, 2], cov=DEFAULT_COVARIANCE, size=n)
    x = np.vstack((a, b))
    y = np.hstack((np.zeros(N), np.ones(N)))
    return x, y


def _cross_entropy_loss(y_pred, y_actual):
    return -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    random_state = 10
    n = 10000

    np.random.seed(random_state)
    X, y = _generate_data(n, random_state)
    fit(X, y, random_state)


