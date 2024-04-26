import math
import random
import pickle

# Set a fixed seed for random to ensure reproducibility
random.seed(0)

def read_csv(filename):
    try:
        lines = open(filename, "r").read().split("\n")
        data = []
        for line in lines[1:]:
            if line.replace(" ", "") != "":
                words = line.split(",")
                data.append([float(words[1]), float(words[2]), float(words[3]), float(words[4]), words[5][1:-1]])
        return data
    
    except Exception as exception:
        print(exception)
        return None

def encode(data, lookup):
    try:
        for row in data:
            row[-1] = lookup[row[-1]]
    except Exception as exception:
        print(exception)

def head(data, rows=5):
    for i in range(rows):
        try:
            print(data[i])
        except:
            break

def shuffle(data):
    try:
        random.shuffle(data)
    except Exception as exception:
        print(exception)

def split_data(data, split_ratio = 0.8):
    try:
        train_size = int(len(data) * split_ratio)
        train_set = data[0:train_size]
        test_set = data[train_size:]
        return train_set, test_set
    
    except Exception as exception:
        print(exception)
        return None, None
 

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
        if verbose:
            print(f'> epoch={epoch+1}, learning_rate={learning_rate}, error={sum_error}')

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


if __name__ == "__main__": 
    data = read_csv("iris.csv")
    lookup = {"setosa": 0, "versicolor": 1, "virginica": 2}
    encode(data, lookup)
    shuffle(data)

    split_ratio = 0.8
    train_set, test_set = split_data(data, split_ratio)

    # Initialize network and train it
    n_inputs = len(train_set[0]) - 1
    n_outputs = len(set(row[-1] for row in train_set))
    learning_rate = 0.1
    n_epochs = 100

    # Specify hidden layer sizes as a list
    hidden_layers = [8, 8]  # Example: Two hidden layers with 8 and 4 neurons respectively

    network = initialize_network(n_inputs, hidden_layers, n_outputs)
    train_network(network, train_set, learning_rate, n_epochs, n_outputs, verbose=False)

    print("Train Accuracy:", check_accuracy(network, train_set))
    print("Test Accuracy:", check_accuracy(network, test_set))

    unclassified = [[5.5, 2.3, 4.0, 1.3, 'versicolor'], [5.7, 2.8, 4.5, 1.3, 'versicolor'], [4.9, 2.4, 3.3, 1.0, 'versicolor']]
    for sample in unclassified:
        print(sample, "=> Predicted Class:", predict(network, sample[:-1]))

    # Save the trained network to a file
    filename = 'ann_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(network, file)
