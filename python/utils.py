import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from model_training import training


def make_model(
        the_tabulatorium: list,
        architecture: int,
        dataset: int,
        version: int
) -> nn.Module:

    model = the_tabulatorium[1][architecture][0]()
    model.load_state_dict(the_tabulatorium[5][architecture][dataset][version][0])

    return model


def update_checkpoint(
        the_tabulatorium: list,
        phase: str,
        location: list
) -> list:
    if phase == 'training':
        architecture, dataset = location
        if dataset < len(the_tabulatorium[2]) - 1:
            checkpoint = ['training', [architecture, dataset + 1]]
        elif architecture < len(the_tabulatorium[1]) - 1:
            checkpoint = ['training', [architecture + 1, 0]]
        else:
            checkpoint = ['explaining', [0, 0, 0, 0]]
    elif phase == 'explaining':
        architecture, dataset, explanation_method, version = location
        if version < len(the_tabulatorium[5][architecture][dataset]) - 1:
            checkpoint = ['explaining', [architecture, dataset, explanation_method, version + 1]]
        elif explanation_method < len(the_tabulatorium[3]) - 1:
            checkpoint = ['explaining', [architecture, dataset, explanation_method + 1, 0]]
        elif dataset < len(the_tabulatorium[2]) - 1:
            checkpoint = ['explaining', [architecture, dataset + 1, 0, 0]]
        elif architecture < len(the_tabulatorium[1]) - 1:
            checkpoint = ['explaining', [architecture + 1, 0, 0, 0]]
        else:
            checkpoint = ['evaluating', [0, 0, 0, 0, 0]]
    elif phase == 'evaluating':
        architecture, dataset, explanation_method, metric, version = location
        if version < len(the_tabulatorium[5][architecture][dataset]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method, metric, version + 1]]
        elif metric < len(the_tabulatorium[4]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method, metric + 1, 0]]
        elif explanation_method < len(the_tabulatorium[3]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method + 1, 0, 0]]
        elif dataset < len(the_tabulatorium[2]) - 1:
            checkpoint = ['evaluation', [architecture, dataset + 1, 0, 0, 0]]
        elif architecture < len(the_tabulatorium[1]) - 1:
            checkpoint = ['evaluation', [architecture + 1, 0, 0, 0, 0]]
        else:
            checkpoint = ['done', []]
    else:
        checkpoint = ['done', []]

    return checkpoint


def train(
        the_tabulatorium: list,
        location: list
) -> list:

    hyperparameters = the_tabulatorium[0][1]
    architecture, dataset = location
    print(f'Training | '
          f'Architecture: {the_tabulatorium[1][architecture][1]} | '
          f'Dataset: {the_tabulatorium[2][dataset][1]}')

    # Train.
    intermediate_versions = training(
        model=the_tabulatorium[1][architecture][0](),
        dataset=the_tabulatorium[2][dataset][0],
        **hyperparameters
    )
    number_versions = len(intermediate_versions)
    the_tabulatorium[5][architecture][dataset] = intermediate_versions
    for explanation_method in range(len(the_tabulatorium[3])):
        the_tabulatorium[6][architecture][dataset][explanation_method] = [None for _ in range(number_versions)]
        for metric in range(len(the_tabulatorium[4])):
            the_tabulatorium[7][architecture][dataset][explanation_method][metric] = \
                [None for _ in range(number_versions)]

    # Update checkpoint.
    checkpoint = update_checkpoint(the_tabulatorium, 'training', location)

    return checkpoint


def explain(
        the_tabulatorium: list,
        location: list
) -> list:

    hyperparameters = the_tabulatorium[0][2]

    architecture, dataset, explanation_method, version = location
    print(f'Explaining | '
          f'Explanation Method: {the_tabulatorium[3][explanation_method][1]} | '
          f'Architecture: {the_tabulatorium[1][architecture][1]} Network | '
          f'Dataset:  {the_tabulatorium[2][dataset][1]} | '
          f'Accuracy: {the_tabulatorium[5][architecture][dataset][version][1] * 100:>4.1f}%')

    # Set the model.
    model = make_model(the_tabulatorium, architecture, dataset, version)

    # Set the explanation method.
    arguments_method_specific = the_tabulatorium[3][explanation_method][2]
    if 'layer' in arguments_method_specific:
        arguments_method_specific['layer'] = model.get_layer()  # fix for the Guided GradCAM method
    method = the_tabulatorium[3][explanation_method][0](model, **arguments_method_specific)
    if 'layer' in arguments_method_specific:
        arguments_method_specific['layer'] = None

    # Set images and labels for explanation.
    images, labels = the_tabulatorium[2][dataset][3]

    # Explain.
    arguments_method_specific = the_tabulatorium[3][explanation_method][3]
    explanations = method.attribute(inputs=images, target=labels, **arguments_method_specific)
    the_tabulatorium[6][architecture][dataset][explanation_method][version] = \
        explanations.sum(axis=1).cpu().detach().numpy()

    # Update checkpoint.
    checkpoint = update_checkpoint(the_tabulatorium, 'explaining', location)

    return checkpoint


def evaluate(
        the_tabulatorium: list,
        location: list,
) -> list:

    hyperparameters = the_tabulatorium[0][3]
    architecture, dataset, explanation_method, metric, version = location
    print(f'Explaining | '
          f'Metric: {the_tabulatorium[4][metric][1]} | '
          f'Explanation Method: {the_tabulatorium[3][explanation_method][1]}'
          f'Architecture: {the_tabulatorium[1][architecture][1]} Network | '
          f'Dataset:  {the_tabulatorium[2][dataset][1]} | '
          f'Accuracy: {the_tabulatorium[5][architecture][dataset][version][1] * 100:>4.1f}%')

    # Set the model.
    model = make_model(the_tabulatorium, architecture, dataset, version)
    model.eval()

    # Get images and labels for explanation.
    images, labels = the_tabulatorium[2][dataset][3]

    # Evaluate.
    """ check what exactly to put here! """
    scores = the_tabulatorium[4][metric][0](**hyperparameters)(
        model=model,
        x_batch=images,
        y_batch=labels,
        a_batch=the_tabulatorium[6][architecture][dataset][explanation_method][metric]
    )
    the_tabulatorium[7][architecture][dataset][explanation_method][metric][version] = scores

    # Update checkpoint.
    checkpoint = update_checkpoint(the_tabulatorium, 'evaluating', location)

    return checkpoint


def generate(
        path: str
) -> None:

    the_tabulatorium, checkpoint = torch.load(path)

    utilities = {
        'training': train,
        'explaining': explain,
        'evaluating': evaluate
    }

    while not checkpoint[0] == 'done':
        checkpoint = utilities[checkpoint[0]](the_tabulatorium, checkpoint[1])
        torch.save([the_tabulatorium, checkpoint], path)
    print('done')

    return


def initialize(
        path: str,
        hyperparameters: list,
        architectures: list,
        datasets: list,
        explanation_methods: list,
        evaluation_metrics: list
) -> None:

    for dataset in datasets:
        dataset_explain = iter(DataLoader(
            dataset=dataset[0](
                root='data',
                train=False,
                download=True,
                transform=ToTensor()
            ),
            **hyperparameters[0]
        )).next()
        dataset.append(dataset_explain)
    versions = [[None for _ in range(len(architectures))] for _ in range(len(datasets))]
    explanations = [[[None for _ in range(len(explanation_methods))] for _ in range(len(architectures))]
                    for _ in range(len(datasets))]
    evaluations = [[[[None for _ in range(len(evaluation_metrics))] for _ in range(len(explanation_methods))]
                    for _ in range(len(architectures))] for _ in range(len(datasets))]

    the_tabulatorium = [
        hyperparameters,
        architectures,
        datasets,
        explanation_methods,
        evaluation_metrics,
        versions,
        explanations,
        evaluations
    ]
    checkpoint = ['training', [0, 0]]

    torch.save([the_tabulatorium, checkpoint], path)

    return
