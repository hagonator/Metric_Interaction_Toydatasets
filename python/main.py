import quantus
import torchvision.datasets
from captum import attr
from torch import nn
import torch

from model_classes import SimpleNet, ConvolutionalNet
from utils import initialize, generate, update_checkpoint

table_model_architectures = [
    [SimpleNet, 'Simple ReLU', 'a fully connected ReLU Network with two hidden layers'],
    [ConvolutionalNet, 'Convolutional ReLU',
     'a convolutional ReLU Network with two convolutional layers with maxpooling and a final fully connected layer'],
]

table_explanation_methods = [
    [attr.Deconvolution, 'Deconvolution', {}, {}],
    [attr.FeatureAblation, 'Feature Ablation', {}, {}],
    [attr.GradientShap, 'Gradient SHAP', {'multiply_by_inputs': False}, {'baselines': torch.zeros([1, 1, 28, 28])}],
    [attr.GuidedBackprop, 'Guided Backpropagation', {}, {}],
    [attr.GuidedGradCam, 'Guided GradCAM', {'layer': None}, {}],    # does not work for simple ReLU
    [attr.InputXGradient, 'Input X Gradient', {}, {}],
    [attr.IntegratedGradients, 'Integrated Gradient', {'multiply_by_inputs': False}, {}],
    [attr.LRP, 'Layer-wise Relevance Propagation', {}, {}],   # rule for flatten!
    [attr.Occlusion, 'Occlusion', {}, {'sliding_window_shapes': (1, 1, 1)}],
    [attr.Saliency, 'Saliency', {}, {}],
    [attr.ShapleyValueSampling, 'Shapley Value Sampling', {}, {}],
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

if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'

table_hyperparameters = [
    # initializing
    {
        'batch_size': 200
    },
    # training
    {
        'batch_size_training': 60,
        'batch_size_testing': 1000,
        'loss_function': nn.CrossEntropyLoss(),
        'goal_accuracy': .9,
        'learning_rate': 1e-4,
        'patience': 20,
        'device': device
    },
    # explaining
    {
        'baselines': 200
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
# )
generate(path)
