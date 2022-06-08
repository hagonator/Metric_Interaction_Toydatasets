from torch import nn

"""
class for the Models
"""


class NeuralNetwork(nn.Module):
    def __init__(self, layers=None):
        """
        :param layers: predefined nn.Sequential(...)
        """
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        if layers:
            self.layers = layers
        else:
            # vanilla option for starting
            self.layers = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

    def forward(self, x):
        x = self.flatten(x)
        layer_output = self.layers(x)
        return layer_output


"""
Simple ReLU

Fully connected ReLU network
Hidden Layers: 1 (512x512)
"""
layers = nn.Sequential(
    nn.Linear(28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
