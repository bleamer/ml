import numpy as np
from pprint import pprint
from random import random

class MLP:



    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(activations, w)

            # calculate activations
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def back_propagate(self, error, verbose=False):

        # dE/dW_i = ( y - a[i+1]) . s'(h[i+1]) . a_i
        # s'(h_[i+1]) = s(h_[i+1])(1-s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        # dE/dW_[i-1] = (y - a_[i+1]) .s'(h_[i+1]). W_i s'(h_i_ .a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]  # ndarray([1,2]) -> ndarray([1],[2])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}".format())
        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0

            for (input, target) in zip(inputs, targets):
                output = self.forward_propagate(input)

                # Calculate the error
                error = target - output

                self.back_propagate(error)

                # Apply gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            print(f"Error: {sum_error/len(inputs)} at epoch{i}")

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return  np.average((target - output)**2)


if __name__ == "__main__":
    # create MLP
    mlp = MLP(2, [5], 1)

    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])

    targets = np.array([i[0] +i[1] for i in inputs])

    mlp.train(inputs, targets, 50, 0.1)
    # Create inputs
    # inputs = np.random.rand(mlp.num_inputs)
    input = np.array([0.1, 0.2])
    target = np.array([0.3])

    input = np.array([0.3, 0.1])
    target = np.array([0,4])

    output = mlp.forward_propagate(input)

    print(f'{input[0]} + {input[1]} is equal to {output[0]}')
    # print results
    # pprint('Inputs are {}'.format(input))
    # pprint('Outputs are {}'.format(output))
