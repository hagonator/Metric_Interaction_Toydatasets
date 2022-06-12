import torch
from captum import attr
import matplotlib.pyplot as plt

from collection_models import *
from collection_datasets import *
from explaination_methods import *

# """
# Integrated Gradients
# """


# def attribute_image(model: NeuralNetwork, algorithm: attr, images: torch.tensor, *args) -> torch.tensor:
#
#     model.zero_grad()
#     predictions = model(images).argmax(axis=1)
#     attributions = algorithm.attribute(images, target=predictions, *args)
#
#     return attributions
#
#
# def explain_ig(model: NeuralNetwork, models_source: str, data: DataLoader, baseline: torch.tensor) -> None:
#     versions = torch.load(models_source)
#     n = len(versions)
#     images, labels = list(data)[0]
#     cols, rows = 3, 3
#     for i in range(0, n, 5):
#         model.load_state_dict(versions[i][0])
#         model.eval()
#         integrated_gradients = attr.IntegratedGradients(model)
#         attributions = attribute_image(model, integrated_gradients, images, baseline)
#         figure = plt.figure(figsize=(10, 10))
#         plt.axis("off")
#         plt.title(f"Number: {i}")
#         for j in range(1, cols * rows + 1):
#             figure.add_subplot(rows, cols, j)
#             plt.axis("off")
#             plt.imshow(attributions[j - 1].squeeze(), cmap='seismic')
#         plt.show()
#     figure = plt.figure(figsize=(10, 10))
#     plt.axis("off")
#     for j in range(1, cols * rows + 1):
#         figure.add_subplot(rows, cols, j)
#         plt.axis("off")
#         plt.imshow(images[j - 1].squeeze(), cmap='gray')
#     plt.show()


# baseline = 0 * torch.normal(.5, 1, [9, 1, 28, 28])

# explain_ig(simple_relu, 'trained_simple_relu.pt', dataloader_MNIST_examples, baseline)
# explain_ig(conv_relu, 'trained_conv_relu.pt', dataloader_FMNIST_examples, baseline)


"""
Generate explanations for some data points for three different states (random, intermediate, final) of the model
"""

# data_explain = FMNIST_test
# size_batch = 4
# dataloader_explain = DataLoader(data_explain, batch_size=size_batch)
# data_explain = iter(dataloader_explain).next()
#
# model = NeuralNetwork(layers_conv_relu)
# model_versions = torch.load('trained_conv_relu.pt')
# n = len(model_versions)
#
# model.load_state_dict(model_versions[n-1][0])
#
# explanations = explain(model, data_explain)
#
# torch.save(explanations, 'explanations_accurate.pt')
#
# model.load_state_dict(model_versions[n//2][0])
#
# explanations = explain(model, data_explain)
#
# torch.save(explanations, 'explanations_inaccurate.pt')
#
# model.load_state_dict(model_versions[0][0])
#
# explanations = explain(model, data_explain)
#
# torch.save(explanations, 'explanations_random.pt')

# explanations_accurate = torch.load('explanations_accurate.pt')
# explanations_inaccurate = torch.load('explanations_inaccurate.pt')
# explanations_random = torch.load('explanations_random.pt')
#
# for method in explanations_accurate:
#     if explanations_accurate[method] is not None:
#         figure = plt.figure(figsize=(16, 16))
#         plt.title(method)
#         plt.axis('off')
#         for i in range(1, 5):
#             figure.add_subplot(4, 4, 0*4+i)
#             plt.axis('off')
#             plt.imshow(data_explain[0][i-1].squeeze(), cmap='gray')
#         for i in range(1, 5):
#             figure.add_subplot(4, 4, 1*4+i)
#             plt.axis('off')
#             plt.imshow(explanations_random[method][i-1].squeeze(), cmap='seismic')
#         for i in range(1, 5):
#             figure.add_subplot(4, 4, 2*4+i)
#             plt.axis('off')
#             plt.imshow(explanations_inaccurate[method][i-1].squeeze(), cmap='seismic')
#         for i in range(1, 5):
#             figure.add_subplot(4, 4, 3*4+i)
#             plt.axis('off')
#             plt.imshow(explanations_accurate[method][i-1].squeeze(), cmap='seismic')
#         plt.show()

"""
Generate explanations for a single image under all model versions
"""

model = NeuralNetwork(layers_simple_relu)
data_explain = iter(DataLoader(MNIST_test, batch_size=2)).next()
model_versions = torch.load('trained_simple_relu.pt')

accuracy = 0
for number_model in model_versions:
    if accuracy + .01 <= model_versions[number_model][1]:
        accuracy = model_versions[number_model][1]
        model.load_state_dict(model_versions[number_model][0])
        explanations = explain(model, data_explain)
        n = len(explanations)
        figure = plt.figure(figsize=(4 * n + 1, 8))
        plt.title(f'Accuracy: {100 * accuracy:4.1f}')
        plt.axis('off')
        figure.add_subplot(2, n + 1, 1)
        plt.title('Original')
        plt.axis('off')
        plt.imshow(data_explain[0][0].squeeze(), cmap='gray')
        figure.add_subplot(2, n + 1, (n + 1) + 1)
        plt.axis('off')
        plt.imshow(data_explain[0][1].squeeze(), cmap='gray')
        i = 1
        for method in explanations:
            i += 1
            figure.add_subplot(2, n + 1, i)
            plt.title(method)
            plt.axis('off')
            plt.imshow(explanations[method][0].squeeze(), cmap='seismic')
            figure.add_subplot(2, n + 1, (n + 1) + i)
            plt.title(method)
            plt.axis('off')
            plt.imshow(explanations[method][1].squeeze(), cmap='seismic')
        plt.show()
