from imagereader import getDataSet
from model import *
import pickle

if __name__ == "__main__":
    print("This program implements an Artificial Neural Network to recognise handwritten digits.")

    print("\nLoading Dataset ...")
    train_set, test_set = getDataSet("../data/trainingSet")
    print("Loaded %d images for training."%len(train_set))
    print("Loaded %d images for testing."%len(test_set))

    # Initialize network and train it
    n_inputs = len(train_set[0]) - 1
    n_outputs = 10
    hidden_layers = [128, 64]
    learning_rate = 0.8
    n_epochs = 20
    print("\nInput Neurons  :", n_inputs)
    print("Hidden Layer   :", hidden_layers)
    print("Output Neurons :", n_outputs)
    print("Learning Rate  :", learning_rate)
    print("Epochs         :", n_epochs)

    network = initialize_network(n_inputs, hidden_layers, n_outputs)
    print("\nInitialized the Neural Network.")
    print("\nStarted model training ...")
    train_network(network, train_set, learning_rate, n_epochs, n_outputs, verbose=True)
    print("Model training finished.")

    print("\nStarted accuracy testing ...")
    print("Train Accuracy:", check_accuracy(network, train_set))
    print("Test  Accuracy:", check_accuracy(network, test_set))

    # Save the trained network to a file
    filename = 'ann_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(network, file)
