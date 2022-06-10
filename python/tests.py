from dataset_collection import *
from model_training import *
from loss_function_collection import *

import matplotlib.pyplot as plt

"""
Generate and save model versions.
"""
goal_accuracy = .8

# torch.save(training(simple_relu, dataloader_MNIST_train, dataloader_MNIST_test, cross_entropy, goal_accuracy,
#                    learning_rate=1e-4), 'trained_simple_relu.pt')

# torch.save(training(conv_relu, dataloader_FMNIST_train, dataloader_FMNIST_test, cross_entropy, goal_accuracy),
#           'trained_conv_relu.pt')

"""
Print for a fixed sample set the predictions of every fifth version of the model.
"""
models = torch.load('trained_simple_relu.pt')
model = NeuralNetwork(layers_simple_relu)
N = len(models)
data_examples_images, data_examples_labels = list(dataloader_MNIST_examples)[0]
for number_model in range(N):
    if number_model % 5 == 0:
        model.load_state_dict(models[number_model][0])
        accuracy = models[number_model][1]
        figure = plt.figure(figsize=(10, 10))
        cols, rows = 3, 3
        with torch.no_grad():
            for i in range(1, cols * rows + 1):
                img, label = data_examples_images[i - 1], data_examples_labels[i - 1]
                prediction = model(img).argmax(1)
                figure.add_subplot(rows, cols, i)
                plt.title(f"True label: {label}\n Prediction: {prediction}")
                plt.axis("off")
                plt.imshow(img.squeeze(), cmap="gray")
            print(accuracy)
        plt.show()
