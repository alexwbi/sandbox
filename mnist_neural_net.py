import numpy as np


class NN(object):

    def __init__(self, neurons_per_layer, random_state=1):
        self.neurons_per_layer = np.array(neurons_per_layer)
        self.num_layers = len(neurons_per_layer)

        np.random.seed(random_state)
        self.weights = self._initial_weights()
        self.biases = self._initial_biases()

    def _initial_weights(self):
        num_neurons_in_consecutive_layers = zip(self.neurons_per_layer[:-1], self.neurons_per_layer[1:])
        return [np.random.standard_normal((y, x)) for x, y in num_neurons_in_consecutive_layers]

    def _initial_biases(self):
        non_initial_layer_neurons = self.neurons_per_layer[1:]
        biases = np.random.standard_normal(non_initial_layer_neurons.sum())
        return np.split(biases, non_initial_layer_neurons[:-1])
