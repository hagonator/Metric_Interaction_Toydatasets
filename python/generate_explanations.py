import torch
from torch.utils.data import DataLoader

from collection_datasets import MNIST_testing
from explaination_methods import explain
from model_classes import SimpleNet, ConvolutionalNet

# Load model versions for both the simple and the convolutional ReLU networks.
simple_models_versions_accuracy = torch.load('simple_relu_versions.pt')
convolutional_models_versions_accuracy = torch.load('convolutional_relu_versions.pt')

# Load samples from the testing dataset for the explanations.
dataloader_explaining = DataLoader(MNIST_testing, batch_size=10, shuffle=False)
dataset_explaining = iter(dataloader_explaining).next()

# Generate explanations for all versions of both model architectures for all selected data points.
# See explanation_methods.py for a list of the used explanation methods.
model = SimpleNet()
for model_version in simple_models_versions_accuracy:
    model.load_state_dict(simple_models_versions_accuracy[model_version][0])
    if len(simple_models_versions_accuracy[model_version]) == 2:
        simple_models_versions_accuracy[model_version].append(explain(model, dataset_explaining))
    else:
        simple_models_versions_accuracy[model_version][2] = explain(model, dataset_explaining)
torch.save(simple_models_versions_accuracy, 'simple_relu_versions.pt')
print('Successfully generated explanations for the simple ReLU network versions.')

model = ConvolutionalNet()
for model_version in convolutional_models_versions_accuracy:
    model.load_state_dict(convolutional_models_versions_accuracy[model_version][0])
    if len(convolutional_models_versions_accuracy[model_version]) == 2:
        convolutional_models_versions_accuracy[model_version].append(explain(model, dataset_explaining))
    else:
        convolutional_models_versions_accuracy[model_version][2] = explain(model, dataset_explaining)
torch.save(convolutional_models_versions_accuracy, 'convolutional_relu_versions.pt')
print('Successfully generated explanations for the convolutional ReLU network versions.')