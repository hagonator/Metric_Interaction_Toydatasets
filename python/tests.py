from model_collection import *
from model_training import *
from dataset_collection import *
from loss_function_collection import *
from accuracy_steps_generator import *
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

accuracy_steps = generate_accuracy_steps(size_steps=3, stop=74)

models = training(simple_relu, dataloader_MNIST_train, dataloader_MNIST_test, cross_entropy, accuracy_steps)
print(models[1])

models_conv = training(conv_relu, dataloader_FMNIST_train, dataloader_FMNIST_test, cross_entropy, accuracy_steps)
print(models_conv[1])

"""
data_examples = [[X, y] for (X, y) in dataloader_MNIST_examples] 
# find the right way to get a fixed subset of MNIST/FMNIST
for model in models:
    figure = plt.figure(figsize=(10, 30))
    cols, rows = 4, 10
    with torch.no_grad():
        for i in range(1, cols * rows + 1):
            # sample_idx = torch.randint(len(data_examples), size=(1,)).item()
            img, label = data_examples[i]
            prediction = model[0](img).argmax(1)
            figure.add_subplot(rows, cols, i)
            plt.title(f"True label: {label}\n Prediction: {prediction}")
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
"""
