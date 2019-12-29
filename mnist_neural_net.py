import numpy as np


class NN(object):

    def __init__(self, neurons_per_layer, random_state=1):
        self.num_layers = len(neurons_per_layer)
        self.neurons_per_layer = np.array(neurons_per_layer)

        np.random.seed(random_state)
        # self.weights = [np.random]
        # self.biases = np.random.randn()

    def _initial_biases(self):
        non_initial_layer_neurons = self.neurons_per_layer[1:]
        biases = np.random.randn((non_initial_layer_neurons.sum(), 1))
        np.random.standard_normal()



