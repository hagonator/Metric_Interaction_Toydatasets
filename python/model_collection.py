import torch
from torch import nn

"""
Class for the Models
"""


class NeuralNetwork(nn.Module):
    def __init__(self, layers: nn.Sequential = None) -> None:
        """
        :param layers: predefined nn.Sequential(...)
        """
        super(NeuralNetwork, self).__init__()
        if layers:
            self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_output = self.layers(x)
        return layer_output


"""
Simple ReLU network

Three fully connected layers
"""
layers_simple_relu = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 2 ** 9),
    nn.ReLU(),
    nn.Linear(2 ** 9, 2 ** 9),
    nn.ReLU(),
    nn.Linear(2 ** 9, 10),
)
simple_relu = NeuralNetwork(layers_simple_relu)

"""
Convolutional ReLU Network

Two convolutional layers with max pooling and a final fully connected layer
"""
layers_conv_relu = nn.Sequential(
    nn.Conv2d(1, 2 ** 4, 5, 1, 2),  # new shape (16x28x28)
    nn.ReLU(),
    nn.MaxPool2d(2),  # new shape (16x14x14)
    nn.Conv2d(2 ** 4, 2 ** 5, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),  # new shape (32x7x7)
    nn.Flatten(),  # new shape (32 * 7 * 7)
    nn.Linear(32 * 7 * 7, 10)
)
conv_relu = NeuralNetwork(layers_conv_relu)

# to be implemented?
"""
Convolutional ReLU Network with (Information) Dropout

Two convolutional layers with max pooling and (Information) dropout and a final fully connected layer

layers_conv_dropout = nn.Sequential(
    nn.Conv2d(1, 2 ** 4, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # (Information Dropout Layer)
    nn.Conv2d(2 ** 4, 2 ** 5, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # (Information Dropout Layer)
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10)
)
conv_dropout = NeuralNetwork(layers_conv_dropout)
"""
