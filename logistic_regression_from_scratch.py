import numpy as np

DEFAULT_COVARIANCE = [[1, 2], [2, 1]]


def fit(X, y, random_seed, include_intercept=True, learning_rate = 1e-5, num_iterations=100000, min_threshold=1e-4):
    np.random.seed(random_seed)
    if include_intercept:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))

    weights = np.random.random((X.shape[1], 1))
    for i in range(num_iterations):
        scores = X.dot(weights)
        y_pred = _sigmoid(scores)
        error = y - y_pred
        gradient = features.T.dot(error)

        delta_weights = learning_rate * gradient
        if delta_weights < min_threshold:
            break

        weights += delta_weights
        if i % 100 == 0:
            print(f'Iteration {i}. Cross entropy loss: {_cross_entropy_loss(y_pred, y)}')

    return weights


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
    random_seed = 10
    n = 10000

    X, y = _generate_data(n, random_seed)
    fit(X, y, random_seed)


