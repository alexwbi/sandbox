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

    def stochastic_gradient_descent(self, training_data, epochs, eta, mini_batch_size):
        """
        :param training_data: training data
        :param epochs: number of epochs
        :param eta: learning rate
        :param mini_batch_size: mini batch size
        """
        for i in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = np.array_split(training_data, mini_batch_size)
            [self._calculate_gradient(mini_batch, eta) for mini_batch in mini_batches]

    def _update_parameters(self, mini_batch, eta):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b_point, delta_w_point = self._backpropagation(x, y)
            delta_b = [b + d_b for b, d_b in zip(delta_b, delta_b_point)]
            delta_w = [w + d_w for w, d_w in zip(delta_w, delta_w_point)]

        batch_size = len(mini_batch)
        # Divide by batch_size to average the changes across all datapoints in the batch
        self.biases = [b - (eta / batch_size) * d_b for b, d_b in zip(self.biases, delta_b)]


    def _backpropagation(self, x, y):
        return x, y  # TODO

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
