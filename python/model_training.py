import torch
from models import *


def training(model: NeuralNetwork, data_train, data_test, accuracy_steps: torch.tensor) -> dict:
    model_versions = []
    accuracy = 0
    epoch_total = 0
    for threshold in accuracy_steps:
        epoch_threshold = 0
        while accuracy < threshold:
            loop_train(model, data_train)
            accuracy = loop_test(model, data_test)
            print(f"Total epochs: {epoch:>5d} |")
        model_versions.append([NeuralNetwork(model.layers), accuracy])
