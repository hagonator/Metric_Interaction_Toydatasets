import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import quantus
import numpy

from model_classes import SimpleNet, ConvolutionalNet

# Load model versions for both the simple and the convolutional ReLU networks already including explanations.
simple_models_versions_accuracy = torch.load('simple_relu_versions.pt')
convolutional_models_versions_accuracy = torch.load('convolutional_relu_versions.pt')

# Set hyperparameters for evaluation metrics.
params_eval = {
    "nr_samples": 10,
    "perturb_radius": 0.1,
    "norm_numerator": quantus.fro_norm,
    "norm_denominator": quantus.fro_norm,
    "perturb_func": quantus.uniform_noise,
    "similarity_func": quantus.difference,
    "img_size": 28,
    "nr_channels": 1,
    "normalise": False,
    "abs": False,
    "disable_warnings": True,
}

# Predefine the evaluation methods to be tested.
metrics = {
    'Maximal Sensitivity': quantus.MaxSensitivity(**params_eval),
    'Average Sensitivity': quantus.AvgSensitivity(**params_eval),
    'Local Lipschitz Estimate': quantus.LocalLipschitzEstimate(**params_eval),
    'Faithfulness Correlation': quantus.FaithfulnessCorrelation(**params_eval),
    'Monotonicity Metric': quantus.MonotonicityNguyen(**params_eval),
    'Infidelity': quantus.Infidelity(**params_eval),
    'Sparseness': quantus.Sparseness(**params_eval),
    'Complexity': quantus.Complexity(**params_eval),
    'Effective Complexity': quantus.EffectiveComplexity(**params_eval)
}

# Evaluate all explanation methods for all versions of both model architectures.
model = SimpleNet()
for model_version in simple_models_versions_accuracy:
    model.load_state_dict(simple_models_versions_accuracy[model_version][0])
    images = simple_models_versions_accuracy[model_version][2]['Originals'][0]
    labels = simple_models_versions_accuracy[model_version][2]['Originals'][1]
    explanation_methods = simple_models_versions_accuracy[model_version][2]
    results = quantus.evaluate(
        metrics=metrics,
        xai_methods=explanation_methods,
        model=model,
        x_batch=images,
        y_batch=labels,
        agg_func=numpy.mean
    )
    if len(simple_models_versions_accuracy[model_version]) == 3:
        simple_models_versions_accuracy[model_version].append(results)
    else:
        simple_models_versions_accuracy[model_version][3] = results
torch.save(simple_models_versions_accuracy, 'simple_relu_versions.pt')
print('Successfully evaluated all explanation methods on all intermediate models of the simple ReLU network '
      'with all the metrics.')

model = ConvolutionalNet()
for model_version in convolutional_models_versions_accuracy:
    model.load_state_dict(convolutional_models_versions_accuracy[model_version][0])
    images = convolutional_models_versions_accuracy[model_version][2]['Originals'][0]
    labels = convolutional_models_versions_accuracy[model_version][2]['Originals'][1]
    explanation_methods = convolutional_models_versions_accuracy[model_version][2]
    results = quantus.evaluate(
        metrics=metrics,
        xai_methods=explanation_methods,
        model=model,
        x_batch=images,
        y_batch=labels,
        agg_func=numpy.mean
    )
    if len(convolutional_models_versions_accuracy[model_version]) == 3:
        convolutional_models_versions_accuracy[model_version].append(results)
    else:
        convolutional_models_versions_accuracy[model_version][3] = results
torch.save(convolutional_models_versions_accuracy, 'convolutional_relu_versions.pt')
print('Successfully evaluated all explanation methods on all intermediate models of the convolutional ReLU network '
      'with all the metrics.')
