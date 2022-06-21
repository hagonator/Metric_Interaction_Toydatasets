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
    """
    initialize a model with given architecture and state dictionary

    :param the_tabulatorium: table containing all relevant information
    :param architecture: location of the model architecture in the_tabulatorium[1]
    :param dataset: location of the dataset it was trained on in the_tabulatorium[2]
    :param version: location of the intermediate version in the tabulatorium[5][architecture][dataset]

    :return: the initialized neural network
    """

    model = the_tabulatorium[1][architecture][0]()
    model.load_state_dict(the_tabulatorium[5][architecture][dataset][version][0])

    return model


def update_checkpoint(
        the_tabulatorium: list,
        phase: str,
        location: list
) -> list:
    """
    given the task that was finished, generate the next tasks coordinates

    :param the_tabulatorium: table containing all relevant information
    :param phase: current phase (training, explaining, evaluating)
    :param location: coordinates of the finished task

    :return: list containing the new task:
        [0]: phase (string)
        [1]: location of the task (list)
    """

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
    """
    train a model and save intermediate versions

    :param the_tabulatorium: table containing all relevant information
    :param location: coordinates in the table for:
        - the model architecture
        - the dataset it shall be trained on

    :return: the new checkpoint, i.e. information on what to do next
    """

    # loading needed information
    hyperparameters = the_tabulatorium[0][1]
    architecture, dataset = location

    # annotation
    print(f'Training | '
          f'Architecture: {the_tabulatorium[1][architecture][1]} | '
          f'Dataset: {the_tabulatorium[2][dataset][1]}')

    # train
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

    # update checkpoint
    checkpoint = update_checkpoint(the_tabulatorium, 'training', location)

    return checkpoint


def explain(
        the_tabulatorium: list,
        location: list
) -> list:
    """
    generate explanations, saving them

    :param the_tabulatorium: table containing all relevant information
    :param location: coordinates in the table for:
        - the model architecture
        - the dataset it was trained on
        - the explanation method to be used
        - the models version

    :return: the new checkpoint, i.e. information on what to do next
    """

    # loading needed information
    hyperparameters = the_tabulatorium[0][2]
    architecture, dataset, explanation_method, version = location

    # annotation
    print(f'Explaining | '
          f'Explanation Method: {the_tabulatorium[3][explanation_method][1]} | '
          f'Architecture: {the_tabulatorium[1][architecture][1]} Network | '
          f'Dataset:  {the_tabulatorium[2][dataset][1]} | '
          f'Accuracy: {the_tabulatorium[5][architecture][dataset][version][1] * 100:>4.1f}%')

    # set the model
    model = make_model(the_tabulatorium, architecture, dataset, version)
    model.eval()

    # set the explanation method (skip explanation if not compatible with model architecture)
    arguments_method_specific = the_tabulatorium[3][explanation_method][2]
    if 'layer' in arguments_method_specific:
        arguments_method_specific['layer'] = model.get_layer()  # fix for the Guided GradCAM method
        if arguments_method_specific['layer'] is None:
            checkpoint = update_checkpoint(the_tabulatorium, 'explaining', location)

            return checkpoint
    method = the_tabulatorium[3][explanation_method][0](model, **arguments_method_specific)
    if 'layer' in arguments_method_specific:
        arguments_method_specific['layer'] = None

    # set images and labels for explanation
    images, labels = the_tabulatorium[2][dataset][3]

    # explain
    arguments_method_specific = the_tabulatorium[3][explanation_method][3]
    explanations = method.attribute(inputs=images, target=labels, **arguments_method_specific)
    the_tabulatorium[6][architecture][dataset][explanation_method][version] = \
        explanations.sum(axis=1).cpu().detach().numpy()

    # update checkpoint
    checkpoint = update_checkpoint(the_tabulatorium, 'explaining', location)

    return checkpoint


def evaluate(
        the_tabulatorium: list,
        location: list,
) -> list:
    """
    evaluate an explanation method on a given model, saving the score

    :param the_tabulatorium: table containing all relevant information
    :param location: coordinates in the table for:
        - the model architecture
        - the dataset it was trained on
        - the used explanation method
        - the evaluation metric to be used
        - the models version

    :return: the new checkpoint, i.e. information on what to do next
    """

    # loading needed information
    hyperparameters = the_tabulatorium[0][3]
    architecture, dataset, explanation_method, metric, version = location

    # annotation
    print(f'Explaining | '
          f'Metric: {the_tabulatorium[4][metric][1]} | '
          f'Explanation Method: {the_tabulatorium[3][explanation_method][1]}'
          f'Architecture: {the_tabulatorium[1][architecture][1]} Network | '
          f'Dataset:  {the_tabulatorium[2][dataset][1]} | '
          f'Accuracy: {the_tabulatorium[5][architecture][dataset][version][1] * 100:>4.1f}%')

    # set the model
    model = make_model(the_tabulatorium, architecture, dataset, version)
    model.eval()

    # get images and labels used for explanation
    images, labels = the_tabulatorium[2][dataset][3]

    # evaluate
    """ check what exactly to put here! """
    scores = the_tabulatorium[4][metric][0](**hyperparameters)(
        model=model,
        x_batch=images.detach().numpy(),
        y_batch=labels.detach().numpy(),
        a_batch=the_tabulatorium[6][architecture][dataset][explanation_method][version]
    )
    the_tabulatorium[7][architecture][dataset][explanation_method][metric][version] = scores

    # update checkpoint
    checkpoint = update_checkpoint(the_tabulatorium, 'evaluating', location)

    return checkpoint


def generate(
        path: str
) -> None:
    """
    generate the content for the previously initialized experimental setup

    :param path: path to the file with initialized experimental setup, also path where generated data will be saved

    :return: nothing
    """

    # load the initialized setup, maybe with advanced progress on generating data
    the_tabulatorium, checkpoint = torch.load(path)

    utilities = {
        'training': train,
        'explaining': explain,
        'evaluating': evaluate
    }

    # iteratively go through the phases of training, explaining and evaluating, saving progress as often as possible
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
    """
    create a collection of lists in a predefined structure for saving experimental data, save it in a file:
        completed upon initialization:
        [0]: hyperparameters (see below)
        [1]: model architectures (see below)
        [2]: datasets (see below)
        [3]: explanation methods (see below)
        [4]: evaluation metrics (see below)

        structure created upon initialization, content to be generated:
        [5]: intermediate model versions
            [model architecture][dataset][intermediate version]
                [0]: the models state dictionary (OrderedDict)
                [1]: accuracy of the intermediate versions (float)
        [6]: explanations
            [model architecture][dataset][explanation method][intermediate version]
                [0]: explanations for a set of fixed inputs (numpy.array)
        [7]: evaluations
            [model architecture][dataset][explanation method][evaluation metric][intermediate version]
                [0]: evaluation for the generated explanations

    :param path: path to the file for saving
    :param hyperparameters: list containing:
        [0]: hyperparameters for initialization (dict)
        [1]: hyperparameters for training (dict)
        [2]: hyperparameters for explaining (dict)
        [3]: hyperparameters for evaluating (dict)
    :param architectures: list containing model architectures in form of lists:
        [0]: the model class (nn.Module)
        [1]: name of the model architecture (string)
        [2]: short description of the model architecture (string)
    :param datasets: list containing datasets in form of lists:
        [0]: the dataset class (torchvision.datasets)
        [1]: name of the dataset (string)
        [2]: short description of the dataset (string)
    :param explanation_methods: list containing explanation methods in form of lists:
        [0]: the explanation method (captum.attr._)
        [1]: name of the explanation method (string)
        [2]: hyperparameters for initializing the method (dict)
        [3]: hyperparameters for attributing (dict)
    :param evaluation_metrics: list containing evaluation metrics in form of lists:
        [0]: the evaluation metric (quantus._)
        [1]: name of the evaluation metric (string)
        [2]: category of the metric (string)

    :return: nothing
    """

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
