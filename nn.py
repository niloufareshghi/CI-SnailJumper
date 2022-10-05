import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)

        in_neurons = layer_sizes[0]
        hidden_neurons = layer_sizes[1]
        out_neurons = layer_sizes[2]
        weights = [np.random.normal(size=(hidden_neurons, in_neurons)),
                   np.random.normal(size=(out_neurons, hidden_neurons))]
        biases = [np.zeros((hidden_neurons, 1), dtype=float), np.zeros((out_neurons, 1), dtype=float)]
        self.weights = weights
        self.biases = biases

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        for i in range(2):
            x = self.activation(self.weights[i] @ x + self.biases[i])
        out = x
        return out
