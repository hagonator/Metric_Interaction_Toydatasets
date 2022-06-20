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
        device: str
) -> list:

    model.to(device)

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
        print(f"Epoch {epoch:>3d} | Starting Accuracy {100*accuracy:>4.1f}% | Versions {len(model_versions):>3d}")
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
    print(f"Epochs {epoch:>3d} | Final Accuracy {100*accuracy:>4.1f}% | Versions {len(model_versions):>3d}")

    return model_versions


def loop_train(
        model: nn.Module,
        dataloader_training: DataLoader,
        loss_function: functional,
        optimizer
) -> None:

    size = len(dataloader_training.dataset)

    for batch, (X, y) in enumerate(dataloader_training):
        prediction = model(X)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"           loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def loop_test(
        model: nn.Module,
        dataloader_testing: DataLoader
) -> float:

    size = len(dataloader_testing.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader_testing:
            prediction = model(X)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    return correct
