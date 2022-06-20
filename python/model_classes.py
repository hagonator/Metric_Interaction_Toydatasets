import torch
from torch import nn


class SimpleNet(nn.Module):

    def __init__(self) -> None:
        super(SimpleNet, self).__init__()
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

    def get_layer(self) -> nn.Module:

        return self.linear1


class ConvolutionalNet(nn.Module):

    def __init__(self) -> None:
        super(ConvolutionalNet, self).__init__()
        self.convolution1 = nn.Conv2d(1, 2 ** 4, 5, 1, 2)  # new shape: [16, 28, 28]
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)  # new shape: [16, 14, 14]
        self.convolution2 = nn.Conv2d(2 ** 4, 2 ** 5, 5, 1, 2)  # new shape: [32, 14, 14]
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)  # new shape: [32, 7, 7]
        self.flatten = nn.Flatten()  # new shape: [32 * 7 * 7]
        self.linear1 = nn.Linear(32 * 7 * 7, 10)

        return

    def forward(self, x: torch.tensor) -> torch.tensor:
        convolution = self.maxpool2(self.relu2(self.convolution2(self.maxpool1(self.relu1(self.convolution1(x))))))
        output = self.linear1(self.flatten(convolution))

        return output

    def get_layer(self) -> nn.Module:

        return self.convolution2
