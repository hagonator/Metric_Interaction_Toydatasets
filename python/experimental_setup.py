import quantus
import torchvision.datasets
from captum import attr
from torch import nn
import torch

from model_classes import SimpleNet, ConvolutionalNet

training_presets = {
    'batch_size_training': 60,
    'batch_size_testing': 1000,
    'loss_function': nn.CrossEntropyLoss,
    'goal_accuracy': .9,
    'learning_rate': 1e-4,
    'patience': 20
}

table_model_architectures = [
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [ConvolutionalNet,
     'Convolutional ReLU',
     'a convolutional ReLU Network with two convolutional layers with maxpooling and a final fully connected layer',
     training_presets],
]

table_datasets = [
    [torchvision.datasets.MNIST, 'MNIST', 'Labeled, handwritten digits'],
    [torchvision.datasets.FashionMNIST, 'FashionMNIST', 'Labeled sketches of fashion items'],
]

table_explanation_methods = [
    [attr.Deconvolution, 'Deconvolution',
     {}, {}],
    [attr.FeatureAblation, 'Feature Ablation',
     {}, {}],
    [attr.GradientShap, 'Gradient SHAP',
     {'multiply_by_inputs': False}, {'baselines': torch.zeros([1, 1, 28, 28])}],
    [attr.GuidedBackprop, 'Guided Backpropagation',
     {}, {}],
    [attr.GuidedGradCam, 'Guided GradCAM',
     {'layer': None}, {}],
    [attr.InputXGradient, 'Input X Gradient',
     {}, {}],
    [attr.IntegratedGradients, 'Integrated Gradient',
     {'multiply_by_inputs': False}, {}],
    [attr.LRP, 'Layer-wise Relevance Propagation',
     {}, {}],  # rule for flatten! (should be handled like the identity?)
    [attr.Occlusion, 'Occlusion',
     {}, {'sliding_window_shapes': (1, 1, 1)}],
    [attr.Saliency, 'Saliency',
     {}, {}],
    [attr.ShapleyValueSampling, 'Shapley Value Sampling',
     {}, {}],
]

table_evaluation_metrics = [
    [quantus.AvgSensitivity, 'Average Sensitivity', 'Robustness', {}],
    [quantus.Infidelity, 'Infidelity', 'Faithfulness', {}],
    [quantus.EffectiveComplexity, 'Effective Complexity', 'Complexity', {}],
    [quantus.ModelParameterRandomisation, 'Model Parameter Randomisation', 'Randomisation', {}]
]