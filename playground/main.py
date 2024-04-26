import pickle
from ann import predict

# Load the trained network from the saved file
filename = 'ann_model.pkl'
with open(filename, 'rb') as file:
    network = pickle.load(file)

unclassified = [[5.5, 2.3, 4.0, 1.3, 'versicolor'], [5.7, 2.8, 4.5, 1.3, 'versicolor'], [4.9, 2.4, 3.3, 1.0, 'versicolor']]
for sample in unclassified:
    print(sample, "=> Predicted Class:", predict(network, sample[:-1]))
