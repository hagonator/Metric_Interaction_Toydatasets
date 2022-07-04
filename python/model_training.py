from copy import deepcopy

import torch
import torchvision
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def training(
        model: nn.Module,
        dataset: torchvision.datasets,
        batch_size_training: int,
        batch_size_testing: int,
        loss_function: functional,
        goal_accuracy: float,
        learning_rate: float,
        patience: int,
) -> list:
    """
    train a model using SGD, saving all intermediate versions exceeding the next full percent of test accuracy

    :param model: some Neural Network, usually randomly initialized
    :param dataset: a torchvision dataset, fitting the Neural Networks expected input
    :param batch_size_training: size of training batches
    :param batch_size_testing: size of test batches
    :param loss_function: the penalizing loss function, usually cross entropy
    :param goal_accuracy: the maximal test accuracy at which to stop the training procedure
    :param learning_rate: the learning rate for the SGD
    :param patience: the maximal number of SGD iterations for exceeding the next full percent of test accuracy

    :return: list containing all saved intermediate versions and their test accuracy
    """

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    loss_function = loss_function()

    dataloader_training = DataLoader(
        dataset=dataset(
            root='data',
            train=True,
            download=True,
            transform=ToTensor()
        ),
        batch_size=batch_size_training,
        shuffle=True
    )
    dataloader_testing = DataLoader(
        dataset=dataset(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        ),
        batch_size=batch_size_testing,
        shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = loop_test(model, dataloader_testing)
    epoch = 1
    model_versions = [[deepcopy(model.state_dict()), accuracy]]
    plateau = 0

    print("--------------------------------------------------")
    while round(accuracy, 2) < goal_accuracy and plateau <= patience:
        print(f"Epoch {epoch:>3d} | Starting Accuracy {100 * accuracy:>4.1f}% | Versions {len(model_versions):>3d}")
        print("--------------------------------------------------")
        epoch += 1
        loop_train(model, dataloader_training, loss_function, optimizer)
        print("--------------------------------------------------")
        accuracy = loop_test(model, dataloader_testing)
        if int(accuracy * 100) > int(model_versions[-1][1] * 100):
            plateau = 0
            model_versions.append([deepcopy(model.state_dict()), accuracy])
        else:
            plateau += 1
    print(f"Epochs {epoch:>3d} | Final Accuracy {100 * accuracy:>4.1f}% | Versions {len(model_versions):>3d}")

    return model_versions


def loop_train(
        model: nn.Module,
        dataloader_training: DataLoader,
        loss_function: functional,
        optimizer
) -> None:
    """
    a sequence of SGD iterations over the whole training set

    :param model: the Neural Network to be trained
    :param dataloader_training: dataloader from the dataset to be trained on with predefined batch size
    :param loss_function: the penalizing loss function, usually cross entropy
    :param optimizer: the optimizer for the SGD

    :return: nothing
    """

    size = len(dataloader_training.dataset)

    for batch, (X, y) in enumerate(dataloader_training):

        # SGD
        prediction = model(X)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # annotation
        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"           loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return


def loop_test(
        model: nn.Module,
        dataloader_testing: DataLoader
) -> float:
    """
    a test on a subset of the dataset to approximate the generalization error

    :param model: the Neural Network to approximate the generalization error for
    :param dataloader_testing: dataloader for the corresponding dataset with predefined batch size

    :return: test accuracy
    """

    size = len(dataloader_testing.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader_testing:
            prediction = model(X)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    return correct


def training_2(
        model: nn.Module,
        dataset: torchvision.datasets,
        batch_size_training: int,
        batch_size_testing: int,
        loss_function: functional,
        goal_accuracies: torch.tensor,
        learning_rate: float,
) -> list:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    loss_function = loss_function()

    dataloader_training = DataLoader(
        dataset=dataset(
            root='data',
            train=True,
            download=True,
            transform=ToTensor()
        ),
        batch_size=batch_size_training,
        shuffle=True
    )
    dataloader_testing = DataLoader(
        dataset=dataset(
            root='data',
            train=False,
            download=True,
            transform=ToTensor()
        ),
        batch_size=batch_size_testing,
        shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = loop_test(model, dataloader_testing)
    epoch = 1
    model_versions = []
    for target_accuracy in goal_accuracies:
        print("--------------------------------------------------")
        while round(accuracy, 2) < target_accuracy:
            print(f"Epoch {epoch:>3d} | Starting Accuracy {100 * accuracy:>4.1f}% | Versions {len(model_versions):>3d}")
            print("--------------------------------------------------")
            epoch += 1
            loop_train(model, dataloader_training, loss_function, optimizer)
            print("--------------------------------------------------")
            accuracy = loop_test(model, dataloader_testing)
        print(f"Epochs {epoch:>3d} | Final Accuracy {100 * accuracy:>4.1f}% | Versions {len(model_versions):>3d}")
        model_versions.append([deepcopy(model.state_dict()), accuracy])

    return model_versions
