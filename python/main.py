import numpy
import quantus
import torch
from torch import nn
from torch.utils.data import DataLoader
from captum import attr

from collection_datasets import MNIST_testing, MNIST_training
from model_classes import SimpleNet, ConvolutionalNet
from model_training import training
from explaination_methods import my_explain

table_model_architectures = [
    [SimpleNet(),
     'A fully connected ReLU Network with two hidden layers.'],
    [ConvolutionalNet(),
     'A convolutional ReLU Network with two convolutional layers with maxpooling and a final fully connected layer'],
]

table_explanation_methods = [
    [attr.Deconvolution, 'Deconvolution', None],
    [attr.FeatureAblation, 'Feature Ablation', None],
    [attr.GradientShap, 'Gradient SHAP', 'multiply_by_input'],
    [attr.GuidedBackprop, 'Guided Backpropagation', None],
    [attr.GuidedGradCam, 'Guided GradCAM', 'layer'],
    [attr.InputXGradient, 'Input X Gradient', None],
    [attr.IntegratedGradients, 'Integrated Gradient', 'multiply_by_input'],
    [attr.LRP, 'Layer-wise Relevance Propagation', None],
    [attr.Occlusion, 'Occlusion', None],
    [attr.Saliency, 'Saliency', None],
    [attr.ShapleyValueSampling, 'Shapley Value Sampling', None],
]

table_evaluation_metrics = [
    [quantus.MaxSensitivity, 'Maximal Sensitivity', 'Robustness'],
    [quantus.AvgSensitivity, 'Average Sensitivity', 'Robustness'],
    [quantus.LocalLipschitzEstimate, 'Local Lipschitz Estimate', 'Robustness'],
    [quantus.FaithfulnessCorrelation, 'Faithfulness Correlation', 'Faithfulness'],
    [quantus.MonotonicityNguyen, 'Monotonicity Metric (Nguyen)', 'Faithfulness'],
    [quantus.Infidelity, 'Infidelity', 'Faithfulness'],
    [quantus.Sparseness, 'Sparseness', 'Complexity'],
    [quantus.Complexity, 'Complexity', 'Complexity'],
    [quantus.EffectiveComplexity, 'Effective Complexity', 'Complexity']
]

path = 'save.pt'

train = True
if train:
    # Train models and save intermediate versions.
    dataloader_training = DataLoader(MNIST_training, batch_size=60)
    dataloader_testing = DataLoader(MNIST_testing, batch_size=1000)
    goal_accuracy_test = .9
    learning_rate = 1e-4
    table_models = [
        training(
            model=table_model_architectures[i][0],
            data_train=dataloader_training,
            data_test=dataloader_testing,
            loss_function=nn.CrossEntropyLoss(),
            goal_accuracy=goal_accuracy_test,
            learning_rate=learning_rate
        )
        for i in range(len(table_model_architectures))]
else:
    table_models = torch.load('data.pt')[1]

explain = True
if explain:
    # Fix a set of images and labels
    dataloader_explaining = DataLoader(MNIST_testing, batch_size=200)
    table_data = torch.tensor(iter(dataloader_explaining).next())

    # Generate explanations for all models with all (compatible) methods using the fixed dataset
    ma = len(table_model_architectures)
    table_explanations = []
    for i in range(ma):
        table_explanations.append([])
        model = table_model_architectures[i][0]
        for j in range(len(table_models[i])):
            table_explanations[i].append([])
            model.load_state_dict(table_models[i][j][0])
            for k in range(len(table_explanation_methods)):
                if table_explanation_methods[k][2] is None:
                    explanations = table_explanation_methods[k][0](model)\
                        .attribute(inputs=table_data[0], targets=table_data[1]).sum(axis=1).cpu().numpy()
                table_explanations[i][j].append(list(explanations))


    # reformat data
    table_data.transpose()
    table_data = list(table_data)
else:
    table_data, table_explanations = torch.load('data.pt')[2, 4]
