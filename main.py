# First import classes along with numpy
from nn_classes import *

np.random.seed(0)

# Initialize inputs
initial_inputs = np.array([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])


################################################################

### N-N with 2 hidden layers
# Create first layer
layer0 = Layer(3, 5)

# Create first hidden layer
layer1 = Layer(5, 6)

# Create second hidden layer
layer2 = Layer(6, 4)

# Create final layer
layer3 = Layer(4, 2)

################################################################

### ACTIVATION FUNCTIONS

# Rectified Linear Units (ReLU), i.e. 0 if x<0 or x if x>0
activation0 = Activation_ReLU()
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()

# Softmax, i.e. exponentiates and normalizes last neuron's inputs
activation3 = Activation_Softmax()


################################################################

# ACTIVATE N-N
layer0.forward(initial_inputs)
activation0.forward(layer0.output)

layer1.forward(activation0.output)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3.forward(activation2.output)
activation3.forward(layer3.output)

print("Results:\n", activation3.output)


################################################################
# EVALUATING THE LOSS

# Target values
y = np.array([0, 0, 1])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation3.output, y)
print("\nLoss: ", loss)
