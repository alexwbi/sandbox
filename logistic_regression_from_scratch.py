import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

DEFAULT_COVARIANCE = [[1, 2], [2, 1]]
np.random.seed(10)


def fit(X, y, learning_rate=1e-5, num_iterations=100000, min_threshold=1e-5):
    weights = np.random.random(X.shape[1])
    for i in range(num_iterations):
        scores = X.dot(weights)
        y_pred = _sigmoid(scores)
        error = y - y_pred
        gradient = X.T.dot(error)
        delta_weights = learning_rate * gradient

        if np.linalg.norm(delta_weights) < min_threshold:
            print(f'Iteration {i}. Cross entropy loss: {_cross_entropy_loss(y_pred, y)}')
            break

        weights += delta_weights
        if i % 100 == 0:
            print(f'Iteration {i}. Cross entropy loss: {_cross_entropy_loss(y_pred, y)}')

    return weights


def predict(x, weights, p_threshold=0.5):
    probability = np.array([_sigmoid(x.dot(weights))])
    return np.where(probability > p_threshold, 1, 0)[0]


def generate_data(n):
    a = np.random.multivariate_normal(mean=[0, 0], cov=DEFAULT_COVARIANCE, size=n)
    b = np.random.multivariate_normal(mean=[3, 3], cov=DEFAULT_COVARIANCE, size=n)
    x = np.vstack((a, b))
    y = np.hstack((np.zeros(n), np.ones(n)))
    return x, y


def _cross_entropy_loss(y_pred, y_actual):
    return -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    X, y = generate_data(10000)
    weights = fit(X, y)
    y_pred = predict(X, weights)
    print(f'Confusion matrix - custom model:\n{confusion_matrix(y, y_pred)}')
    print(f'Accuracy - custom_model: {accuracy_score(y, y_pred)}')

    sklearn_logistic_regression = LogisticRegression(fit_intercept=False, C=1e10, solver='lbfgs')
    sklearn_logistic_regression.fit(X, y)
    y_pred_sklearn = sklearn_logistic_regression.predict(X)
    print(f'Confusion matrix - scikit-learn:\n{confusion_matrix(y, y_pred_sklearn)}')
    print(f'Accuracy - scikit-learn: {accuracy_score(y, y_pred_sklearn)}')
