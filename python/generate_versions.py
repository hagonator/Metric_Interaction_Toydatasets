import torch
from torch import nn
from torch.utils.data import DataLoader

from collection_datasets import MNIST_testing, MNIST_training
from model_classes import SimpleNet, ConvolutionalNet
from model_training import training

# Initialize a simple and a convolutional ReLU neural network.
simple_relu = SimpleNet()
convolutional_relu = ConvolutionalNet()

# Load and save the MNIST dataset in a training and testing split.
# Initialize related data loaders.
dataloader_training = DataLoader(MNIST_training, batch_size=60, shuffle=True)
dataloader_testing = DataLoader(MNIST_testing, batch_size=1000, shuffle=True)

# Set hyperparameters for training.
goal_accuracy_test = .9
learning_rate = 1e-4

# Run training for both the simple and the convolutional ReLU neural networks.
# Save the intermediate versions in separate files for both architectures.
simple_models_versions_accuracy = training(simple_relu, dataloader_training, dataloader_testing,
                                           nn.CrossEntropyLoss(), goal_accuracy_test, learning_rate)
torch.save(simple_models_versions_accuracy, 'simple_relu_versions.pt')
print('Successfully saved versions of the simple ReLU network.')

convolutional_models_versions_accuracy = training(convolutional_relu, dataloader_training, dataloader_testing,
                                                  nn.CrossEntropyLoss(), goal_accuracy_test, learning_rate)
torch.save(convolutional_models_versions_accuracy, 'convolutional_relu_versions.pt')
print('Successfully saved versions of the convolutional ReLU network.')
