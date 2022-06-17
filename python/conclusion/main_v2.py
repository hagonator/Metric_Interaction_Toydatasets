import quantus
import torchvision.datasets
from captum import attr
from torch import nn
import torch

from model_classes_v2 import SimpleNet, ConvolutionalNet
from utils import initialize, generate

table_model_architectures = [
    [SimpleNet(),
     'a fully connected ReLU Network with two hidden layers'],
    [ConvolutionalNet(),
     'a convolutional ReLU Network with two convolutional layers with maxpooling and a final fully connected layer'],
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
    [quantus.EffectiveComplexity, 'Effective Complexity', 'Complexity'],
]

table_datasets = [
    [torchvision.datasets.MNIST, 'MNIST', 'Labeled, handwritten digits'],
    [torchvision.datasets.FashionMNIST, 'FashionMNIST', 'Labeled sketches of fashion items'],
]

table_hyperparameters = [
    # training
    {
        'batch_size_training': 60,
        'batch_size_testing': 1000,
        'loss_function': nn.CrossEntropyLoss(),
        'goal_accuracy': .9,
        'learning_rate': 1e-4
    },
    # explaining
    {
        'batch_size': 200
    },
    # evaluating
    {

    }
]

path = 'test.pt'
# initialize(
#    path=path,
#    hyperparameters=table_hyperparameters,
#    architectures=table_model_architectures,
#    datasets=table_datasets,
#    explanation_methods=table_explanation_methods,
#    evaluation_metrics=table_evaluation_metrics
#)
# generate('test.pt')
print(torch.load(path)[1])