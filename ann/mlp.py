import numpy as np
from pprint import pprint
class MLP:

    def __init__(self, num_inputs=3, num_hidden=[3,5], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden

        layers = [self.num_inputs] + self.num_hidden +[self.num_outputs]

        #initialize random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)
    
    def forward_propagate(self, inputs):

        activations = inputs
        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)


            # calculate activations
            activations = self._sigmoid(net_inputs)
        return activations
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # create MLP
    mlp = MLP()

    #Create inputs
    inputs = np.random.rand(mlp.num_inputs)

    # Forward propagation
    outputs = mlp.forward_propagate(inputs)

    #print results
    pprint('Inputs are {}'.format(inputs))
    pprint('Outputs are {}'.format(outputs))
