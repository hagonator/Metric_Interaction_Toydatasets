import torch
from model_collection import *
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn import functional
import matplotlib.pyplot as plt


def training(model: NeuralNetwork, data_train: DataLoader, data_test: DataLoader, loss_function: functional,
             accuracy_steps: torch.tensor, learning_rate: float = 1e-3, give_examples: bool = False) -> list:
    learning_rate = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = loop_test(model, data_test, loss_function)
    model_versions = [[NeuralNetwork(model.layers), accuracy]]
    if give_examples:
        generate_examples(model, data_test.dataset)
    epoch_total = 0

    for i, threshold in enumerate(accuracy_steps):
        epoch_threshold = 0
        while accuracy < threshold:
            epoch_total += 1
            epoch_threshold += 1
            loop_train(model, data_train, loss_function, learning_rate)
            accuracy = loop_test(model, data_test, loss_function)
            print(
                f"Total epochs: {epoch_total:>3d} | Threshold epochs: {epoch_threshold:>3d} | Accuracy: {accuracy:>4f} "
                f"| Current threshold: {threshold:>4f}\n")
        model_versions.append([NeuralNetwork(model.layers), accuracy])
        if give_examples:
            generate_examples(model, data_test.dataset)

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
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def loop_test(model: NeuralNetwork, data_test: DataLoader, loss_function: functional) -> float:
    size = len(data_test.dataset)
    number_batches = len(data_test)
    loss_test, correct = 0, 0

    with torch.no_grad():
        for X, y in data_test:
            prediction = model(X)
            loss_test += loss_function(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    loss_test /= number_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss_test:>8f} \n")

    return correct


def generate_examples(model: NeuralNetwork, data_examples: DataLoader) -> None:
    figure = plt.figure(figsize=(10, 30))
    cols, rows = 4, 10
    with torch.no_grad():
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(data_examples), size=(1,)).item()
            img, label = data_examples[sample_idx]
            prediction = int(model(img).argmax(1))
            figure.add_subplot(rows, cols, i)
            plt.title(f"True label: {label}\n Prediction: {prediction}")
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
