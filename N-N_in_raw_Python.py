import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


np.random.seed(0)

# Initialize inputs
initial_inputs = np.array([[1, 2, 3, 2.5], [2, 1, 5, 9.5]])

### N-N with 2 hidden layers
# Create first layer
layer0 = Layer(4, 5)

# Create first hidden layer
layer1 = Layer(5, 6)

# Create second hidden layer
layer2 = Layer(6, 4)

# Create final layer
layer3 = Layer(4, 2)

# Activation Functions (Rectified Linear, i.e. 0 if x<0 or x if x>0)
activation0 = Activation_ReLU()
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()


# Activate N-N
layer0.forward(initial_inputs)
activation0.forward(layer0.output)

layer1.forward(activation0.output)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3.forward(activation2.output)
activation3.forward(layer3.output)

print(activation3.output)

