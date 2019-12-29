import numpy as np


class NN(object):

    def __init__(self, neurons_per_layer, random_state=1):
        self.neurons_per_layer = np.array(neurons_per_layer)
        self.num_layers = len(neurons_per_layer)

        np.random.seed(random_state)
        self.weights = self._initial_weights()
        self.biases = self._initial_biases()

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(a.dot(b) + w)
        return a

    def stochastic_gradient_descent(self, X, epochs, eta, mini_batch_size):
        """
        :param X: training data
        :param epochs: number of epochs
        :param eta: learning rate
        :param mini_batch_size: mini batch size
        """
        for i in range(epochs):
            np.random.shuffle(X)
            mini_batches = np.array_split(X, mini_batch_size)
            [self._calculate_gradient(mini_batch, eta) for mini_batch in mini_batches]

    def _initial_weights(self):
        num_neurons_in_consecutive_layers = zip(self.neurons_per_layer[:-1], self.neurons_per_layer[1:])
        return [np.random.standard_normal((y, x)) for x, y in num_neurons_in_consecutive_layers]

    def _initial_biases(self):
        non_initial_layer_neurons = self.neurons_per_layer[1:]
        biases = np.random.standard_normal(non_initial_layer_neurons.sum())
        return np.reshape(np.split(biases, non_initial_layer_neurons[:-1]), (1, -1))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
