import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Initialize inputs
initial_inputs = np.array([[1, 2, 3, 2.5]])

### N-N with 2 hidden layers
# Create first layers
layer0 = Layer(4, 5)

# Create first hidden layer
layer1 = Layer(5, 6)

# Create second hidden layer
layer2 = Layer(6, 4)

# Create final layer
layer3 = Layer(4, 2)

# Activate N-N
layer0.forward(initial_inputs)
layer1.forward(layer0.output)
layer2.forward(layer1.output)
layer3.forward(layer2.output)

print(layer3.output)

