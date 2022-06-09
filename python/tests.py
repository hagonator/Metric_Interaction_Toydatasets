from model_collection import *
from model_training import *
from dataset_collection import *
from loss_function_collection import *
from accuracy_steps_generator import *
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

accuracy_steps = generate_accuracy_steps(size_steps=3, stop=83)

models = training(simple_relu, dataloader_FMNIST_train, dataloader_FMNIST_test, cross_entropy, accuracy_steps,
                  give_examples=True)
print(models[:, 1])
