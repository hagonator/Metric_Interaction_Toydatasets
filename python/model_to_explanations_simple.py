import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import quantus

from model_training import *
from explaination_methods import *


class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super(SimpleNet).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 2 ** 9)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2 ** 9, 2 ** 9)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(2 ** 9, 10)

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(self.flatten(x))))))

        return output


class ConvolutionalNet(nn.Module):

    def __init__(self) -> None:
        super(ConvolutionalNet).__init__()
        self.convolution1 = nn.Conv2d(1, 28 * 28, 2 ** 4, 5, 1, 2)  # new shape: [16, 28, 28]
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # new shape: [16, 14, 14]
        self.convolution2 = nn.Conv2d(2 ** 4, 2 ** 5, 5, 1, 2)  # new shape: [32, 14, 14]
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # new shape: [32, 7, 7]
        self.flatten = nn.Flatten() # new shape: [32 * 7 * 7]
        self.linear1 = nn.Linear(32 * 7 * 7, 10)

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        convolution = self.maxpool2(self.relu2(self.convolution2(self.maxpool1(self.relu1(self.convolution1(x))))))
        output = self.linear1(self.flatten(convolution))

        return output


# Initialize a simple and a convolutional ReLU neural network.
simple_relu = SimpleNet()
convolutional_relu = ConvolutionalNet()

# Load and save the MNIST dataset in a training and testing split.
# Initialize related data loaders.
dataset_training = torchvision.datasets.MNIST(
    root='data',
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor(),
)
dataloader_training = DataLoader(dataset_training, batch_size=60, shuffle=True)
dataset_testing = torchvision.datasets.MNIST(
    root='data',
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)
dataloader_testing = DataLoader(dataset_testing, batch_size=100, shuffle=True)

# Set hyperparameters for training.
goal_accuracy_test = .9
learning_rate = 1e-4

# Run training for both the simple and the convolutional ReLU neural networks.
# Save the intermediate versions in separate files for both architectures.
simple_models_versions_accuracy = training(simple_relu, dataloader_training, dataloader_testing, nn.CrossEntropyLoss(),
                                           goal_accuracy_test, learning_rate)
torch.save(simple_models_versions_accuracy, 'simple_relu_versions.pt')
print('Successfully saved versions of the simple ReLU network.')
convolutional_models_versions_accuracy = training(convolutional_relu, dataloader_training, dataloader_testing,
                                                  nn.CrossEntropyLoss(), goal_accuracy_test, learning_rate)
torch.save(convolutional_models_versions_accuracy, 'convolutional_relu_versions.pt')
print('Successfully saved versions of the convolutional ReLU network.')

# Load samples from the testing dataset for the explanations.
dataloader_explaining = DataLoader(dataset_testing, batch_size=10, shuffle=False)
dataset_explaining = iter(dataloader_explaining).next()

# Generate explanations for all versions of both model architectures for all selected data points.
# See explanation_methods.py for a list of the used explanation methods.
model = SimpleNet()
for model_version in simple_models_versions_accuracy:
    model.load_state_dict(simple_models_versions_accuracy[model_version][0])
    simple_models_versions_accuracy[model_version].append(explain(model, dataset_explaining))
torch.save(simple_models_versions_accuracy, 'simple_relu_versions.pt')
print('Successfully generated explanations for the simple ReLU network versions.')

model = ConvolutionalNet()
for model_version in convolutional_models_versions_accuracy:
    model.load_state_dict(convolutional_models_versions_accuracy[model_version][0])
    convolutional_models_versions_accuracy[model_version].append(explain(model, dataset_explaining))
torch.save(convolutional_models_versions_accuracy, 'convolutional_relu_versions.pt')
print('Successfully generated explanations for the convolutional ReLU network versions.')

# Predefine the evaluation methods to be tested.
metrics = {

}

# Evaluate all explanation methods for all versions of both model architectures.
model = SimpleNet()
for model_version in simple_models_versions_accuracy:
    model.load_state_dict(simple_models_versions_accuracy[model_version][0])
    images = dataset_explaining[0]
    labels = dataset_explaining[1]
    explanation_methods = simple_models_versions_accuracy[model_version][2]
