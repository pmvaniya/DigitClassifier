import math
import random

# Set a fixed seed for random to ensure reproducibility
random.seed(0)

def initialize_network(n_inputs, hidden_layers, n_outputs):
    network = []
    
    # Add first hidden layer
    hidden_layer1 = [{'weights': [random.uniform(-0.5, 0.5) for _ in range(n_inputs + 1)]} for _ in range(hidden_layers[0])]
    network.append(hidden_layer1)
    
    # Add additional hidden layers
    for i in range(1, len(hidden_layers)):
        hidden_layer = [{'weights': [random.uniform(-0.5, 0.5) for _ in range(hidden_layers[i - 1] + 1)]} for _ in range(hidden_layers[i])]
        network.append(hidden_layer)
    
    # Add output layer
    output_layer = [{'weights': [random.uniform(-0.5, 0.5) for _ in range(hidden_layers[-1] + 1)]} for _ in range(n_outputs)]
    network.append(output_layer)
    
    return network

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

def sigmoid(activation):
    return 1.0 / (1.0 + math.exp(-activation))

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j, neuron in enumerate(layer):
                errors.append(expected[j] - neuron['output'])
        for j, neuron in enumerate(layer):
            neuron['delta'] = errors[j] * neuron['output'] * (1.0 - neuron['output'])

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']

def train_network(network, train, learning_rate, n_epoch, n_outputs, verbose=False):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0] * n_outputs
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
        print(f'> epoch = {epoch+1}, learning_rate = {learning_rate}, error = {sum_error}')
        if learning_rate > 0.05:
            learning_rate -= 0.05

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def check_accuracy(network, dataset):
    predictions = []
    actual = [row[-1] for row in dataset]
    for row in dataset:
        prediction = predict(network, row)
        predictions.append(prediction)
    return accuracy_metric(actual, predictions)
