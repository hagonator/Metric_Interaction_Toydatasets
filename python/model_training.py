from copy import deepcopy

from torch.nn import functional
from torch.utils.data import DataLoader

from collection_models import *

"""
Training procedure saving all intermediate versions (in between training loops) of the model.
Functions for a single training loop / a single test loop are outsourced in separate functions
"""


def training(model: NeuralNetwork, data_train: DataLoader, data_test: DataLoader, loss_function: functional,
             goal_accuracy: float, learning_rate: float = 1e-3) -> dict:

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = loop_test(model, data_test, loss_function)
    epoch = 0
    model_versions = dict()

    print("-----------------------------------")
    while accuracy < goal_accuracy:
        model_versions[epoch] = [deepcopy(model.state_dict()), accuracy]
        print(f"Epoch {epoch:>3d} | Starting Accuracy {100*accuracy:>4.1f}%")
        print("-----------------------------------")
        epoch += 1
        loop_train(model, data_train, loss_function, optimizer)
        print("-----------------------------------")
        accuracy = loop_test(model, data_test, loss_function)
    print(f"Epochs {epoch:>3d} | Final Accuracy {100*accuracy:>4.1f}%")

    return model_versions


def loop_train(model: NeuralNetwork, data_train: DataLoader, loss_function: functional, optimizer) -> None:

    size = len(data_train.dataset)

    for batch, (X, y) in enumerate(data_train):
        prediction = model(X)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"   loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def loop_test(model: NeuralNetwork, data_test: DataLoader, loss_function: functional) -> float:

    size = len(data_test.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in data_test:
            prediction = model(X)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    correct /= size

    return correct

