import torch

from collection_models import *
# from model_to_explanations_simple import *
from collection_datasets import *
from torch.utils.data import DataLoader
from torchvision import datasets
from captum import attr

"""
Implement procedure for generating explanations

explanation methods (most of the primary attribution methods implemented by captum):
    - Integrated Gradient 
    - Smoothed Integrated Gradient 
    - Gradient SHAP
    - DeepLIFT
    - DeepLIFT SHAP
    - Saliency
    - Input X Gradient
    - Guided Backpropagation and Deconvolution
    - Guided GradCAM
    - Feature Ablation
    - Feature Permutation
    - Occlusion
    - Shapley Value Sampling

have a close look at each and every method, check compatibility
"""

def my_explain(model: nn.Module, data_explain: torch.tensor) -> dict:
    return explain(model, data_explain)

def explain(model: nn.Module, data_explain: torch.tensor) -> dict:
    explanations = {
        'Originals': data_explain.cpu().numpy(),
        'Integrated Gradient': None,
        'Smoothed Integrated Gradient': None,
        'Gradient SHAP': None,
        'DeepLIFT': None,
        # 'DeepLIFT SHAP': None, # needs batch size > 1?
        'Saliency': None,
        'Guided Backpropagation': None,
        'Deconvolution': None,
        # 'Layer-wise Relevance Propagation': None, # fix problem with the flatten-module
        # 'Guided GradCAM': None, # implement differentiation between 'conv' and 'simple' to use this
        'Feature Ablation': None,
        'Occlusion': None,
        'Shapley Value Sampling': None,
        'Input X Gradient': None
    }

    images = data_explain[0]
    labels = data_explain[1]
    baseline = torch.zeros_like(images)  # redundant for now

    explanations_ig = attr.IntegratedGradients(model, multiply_by_inputs=False).attribute(inputs=images, target=labels,
                                                                                          baselines=baseline)
    explanations['Integrated Gradient'] = explanations_ig.sum(axis=1).cpu().numpy()

    explanations_sig = attr.NoiseTunnel(attr.IntegratedGradients(model, multiply_by_inputs=False))\
        .attribute(inputs=images, target=labels, baselines=baseline, nt_samples=10)
    explanations['Smoothed Integrated Gradient'] = explanations_sig.sum(axis=1).cpu().numpy()

    explanations_gs = attr.GradientShap(model, multiply_by_inputs=False).attribute(inputs=images, target=labels,
                                                                                   baselines=baseline)
    explanations['Gradient SHAP'] = explanations_gs.sum(axis=1).cpu().numpy()

    explanations_dl = attr.DeepLift(model, multiply_by_inputs=False).attribute(inputs=images, target=labels,
                                                                               baselines=baseline)
    explanations['DeepLIFT'] = explanations_dl.sum(axis=1).cpu().detach().numpy()

    # explanations_dls = attr.DeepLiftShap(model).attribute(inputs=images, target=labels, baselines=baseline)
    # explanations['DeepLIFT SHAP'] = explanations_dls.sum(axis=1).cpu().detach().numpy()

    explanations_s = attr.Saliency(model).attribute(inputs=images, target=labels, abs=False)
    explanations['Saliency'] = explanations_s.sum(axis=1).cpu().numpy()

    explanations_gb = attr.GuidedBackprop(model).attribute(inputs=images, target=labels)
    explanations['Guided Backpropagation'] = explanations_gb.sum(axis=1).cpu().numpy()

    explanations_dc = attr.Deconvolution(model).attribute(inputs=images, target=labels)
    explanations['Deconvolution'] = explanations_dc.sum(axis=1).cpu().numpy()

    # explanations_lrp = attr.LRP(model).attribute(inputs=images, target=labels)
    # explanations['Guided Backpropagation'] = explanations_lrp.sum(axis=1).cpu().numpy()

    # CNN only
    # explanations_ggc = attr.GuidedGradCam(model, layer=None).attribute(inputs=images, target=labels)
    # explanations['Guided GradCAM'] = explanations_ggc.sum(axis=1).cou().numpy()

    explanations_fa = attr.FeatureAblation(model).attribute(inputs=images, target=labels, baselines=baseline)
    explanations['Feature Ablation'] = explanations_fa.sum(axis=1).cpu().numpy()

    explanations_o = attr.Occlusion(model).attribute(inputs=images, target=labels, baselines=baseline,
                                                     sliding_window_shapes=(1, 1, 1))
    explanations['Occlusion'] = explanations_o.sum(axis=1).cpu().numpy()

    explanations_svs = attr.ShapleyValueSampling(model).attribute(inputs=images, target=labels, baselines=baseline)
    explanations['Shapley Value Sampling'] = explanations_svs.sum(axis=1).cpu().numpy()

    explanations_ixg = attr.InputXGradient(model).attribute(inputs=images, target=labels)
    explanations['Input X Gradient'] = explanations_ixg.sum(axis=1).cpu().detach().numpy()

    return explanations
