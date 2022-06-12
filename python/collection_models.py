import torch
from torch import nn

"""
Class for the Models
"""


class NeuralNetwork(nn.Module):
    def __init__(self, layers: nn.Sequential = None, last_conv_layer: nn.Module = None) -> None:
        """
        :param layers: predefined nn.Sequential(...)
        """
        super(NeuralNetwork, self).__init__()
        if layers:
            self.layers = layers
        if  last_conv_layer:
            self.last_conv_layer = last_conv_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_output = self.layers(x)
        return layer_output


"""
Auxiliary Linear Network 

One fully connected layer without activation function
"""
layers_auxiliary_linear = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
)
auxiliary_linear = NeuralNetwork(layers_auxiliary_linear)

"""
Simple ReLU Network

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
last_conv_layer = nn.Conv2d(2 ** 4, 2 ** 5, 5, 1, 2)
layers_conv_relu = nn.Sequential(
    nn.Conv2d(1, 2 ** 4, 5, 1, 2),  # new shape (16x28x28)
    nn.ReLU(),
    nn.MaxPool2d(2),  # new shape (16x14x14)
    last_conv_layer,
    nn.ReLU(),
    nn.MaxPool2d(2),  # new shape (32x7x7)
    nn.Flatten(),  # new shape (32 * 7 * 7)
    nn.Linear(32 * 7 * 7, 10)
)
conv_relu = NeuralNetwork(layers_conv_relu, last_conv_layer)

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
