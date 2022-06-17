from torchvision import datasets
from torchvision.transforms import ToTensor

"""
The standard MNIST dataset
Handwritten digits 0-9, grayscale

Instance Format: torch tensor, (28x28,1), float in [0,1]
Training instances: 60k
Test instances: 10k
"""
MNIST_training = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
MNIST_testing = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


"""
The FashionMNIST dataset
Sketches of cloths, 10 classes 0-9, grayscale

Instance Format: torch tensor, (28x28,1), float in [0,1]
Training instances: 60k
Test instances: 10k
"""

FMNIST_training = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
FMNIST_testing = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
