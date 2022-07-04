import quantus
import torchvision.datasets
from captum import attr
from torch import nn
import torch

from model_classes import SimpleNet, ConvolutionalNet

"""
First experiments
"""

training_presets = {
    'batch_size_training': 60,
    'batch_size_testing': 1000,
    'loss_function': nn.CrossEntropyLoss,
    'goal_accuracies': torch.tensor([.6, .7, .8]),
    'learning_rate': 1e-4,
}

table_model_architectures = [
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
    [SimpleNet,
     'Simple ReLU',
     'a fully connected ReLU Network with two hidden layers',
     training_presets],
]

table_datasets = [
    [torchvision.datasets.MNIST, 'MNIST', 'Labeled, handwritten digits'],
]

table_explanation_methods = [
    ['GradientShap'],
    ['IntegratedGradients'],
    ['InputXGradient'],
    ['Saliency'],
    ['FeatureAblation'],
    ['GradCam']
]

evaluation_presets = {
    'display_progressbar': True,
    'disable_warnings': True,
}

table_evaluation_metrics = [
    [quantus.AvgSensitivity, 'Average Sensitivity, sample size=100', 'Robustness',
     {**evaluation_presets, 'nr_samples': 100}],
    [quantus.Infidelity, 'Infidelity', 'Faithfulness', evaluation_presets],
    [quantus.EffectiveComplexity, 'Effective Complexity, epsilon=1e-5', 'Complexity',
     {**evaluation_presets, 'eps': 1e-5}],
    [quantus.EffectiveComplexity, 'Effective Complexity, epsilon=1e-4', 'Complexity',
     {**evaluation_presets, 'eps': 1e-4}],
    [quantus.EffectiveComplexity, 'Effective Complexity, epsilon=1e-3', 'Complexity',
     {**evaluation_presets, 'eps': 1e-3}],
    [quantus.EffectiveComplexity, 'Effective Complexity, epsilon=1e-2', 'Complexity',
     {**evaluation_presets, 'eps': 1e-2}],
]

"""
Second experiments
"""
