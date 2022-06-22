import torch
from torch import nn

from model_training import training


def initialize(
        path: str,
        architectures: list,
        datasets: list,
        explanation_methods: list,
        evaluation_metrics: list
) -> None:
    """
    create a collection of lists in a predefined structure for saving experimental data, save it in a file:
        completed upon initialization:
        [0]: model architectures (see below)
        [1]: datasets (see below)
        [2]: explanation methods (see below)
        [3]: evaluation metrics (see below)

        structure created upon initialization, content to be generated:
        [4]: intermediate model versions
            [model architecture][dataset][intermediate version]
                [0]: the models state dictionary (OrderedDict)
                [1]: accuracy of the intermediate versions (float)
        [5]: evaluations
            [model architecture][dataset][explanation method][evaluation metric][intermediate version]
                [0]: evaluation for the generated explanations

    :param path: path to the file for saving
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

    versions = [[None for _ in range(len(architectures))] for _ in range(len(datasets))]
    explanations = [[[None for _ in range(len(explanation_methods))] for _ in range(len(architectures))]
                    for _ in range(len(datasets))]
    evaluations = [[[[None for _ in range(len(evaluation_metrics))] for _ in range(len(explanation_methods))]
                    for _ in range(len(architectures))] for _ in range(len(datasets))]

    the_tabulatorium = [
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
        'evaluating': evaluate
    }

    # iteratively go through the phases of training, explaining and evaluating, saving progress as often as possible
    while not checkpoint[0] == 'done':
        checkpoint = utilities[checkpoint[0]](the_tabulatorium, checkpoint[1])
        torch.save([the_tabulatorium, checkpoint], path)
    print('done')

    return


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
    architecture, dataset = location
    hyperparameters = the_tabulatorium[0][architecture][3]

    # annotation
    print(f'Training | '
          f'Architecture: {the_tabulatorium[0][architecture][1]} | '
          f'Dataset: {the_tabulatorium[1][dataset][1]}')

    # train
    intermediate_versions = training(
        model=the_tabulatorium[0][architecture][0](),
        dataset=the_tabulatorium[1][dataset][0],
        **hyperparameters
    )
    number_versions = len(intermediate_versions)
    the_tabulatorium[4][architecture][dataset] = intermediate_versions
    for explanation_method in range(len(the_tabulatorium[2])):
        the_tabulatorium[5][architecture][dataset][explanation_method] = [None for _ in range(number_versions)]
        for metric in range(len(the_tabulatorium[3])):
            the_tabulatorium[6][architecture][dataset][explanation_method][metric] = \
                [None for _ in range(number_versions)]

    # update checkpoint
    checkpoint = update_checkpoint(the_tabulatorium, 'training', location)

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
    architecture, dataset, explanation_method, metric, version = location
    metric_hyperparameters = the_tabulatorium[3][metric][3]

    # annotation
    print(f'Explaining | '
          f'Metric: {the_tabulatorium[3][metric][1]} | '
          f'Explanation Method: {the_tabulatorium[2][explanation_method][1]}'
          f'Architecture: {the_tabulatorium[0][architecture][1]} Network | '
          f'Dataset:  {the_tabulatorium[1][dataset][1]} | '
          f'Accuracy: {the_tabulatorium[4][architecture][dataset][version][1] * 100:>4.1f}%')

    # set the model
    model = make_model(the_tabulatorium, architecture, dataset, version)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()

    # get images and labels used for explanation
    images, labels = the_tabulatorium[1][dataset][3]

    # set explanation method
    explanation_hyperparameters = the_tabulatorium[3][explanation_method][3]
    method = the_tabulatorium[3][explanation_method][0](**explanation_hyperparameters)

    # evaluate
    explanation_hyperparameters = the_tabulatorium[3][explanation_method][3]
    scores = the_tabulatorium[3][metric][0](**metric_hyperparameters)(
        model=model,
        x_batch=images.detach().numpy(),
        y_batch=labels.detach().numpy(),
        explain_func=method,
        **explanation_hyperparameters
    )
    the_tabulatorium[6][architecture][dataset][explanation_method][metric][version] = scores

    # update checkpoint
    checkpoint = update_checkpoint(the_tabulatorium, 'evaluating', location)

    return checkpoint


def make_model(
        the_tabulatorium: list,
        architecture: int,
        dataset: int,
        version: int
) -> nn.Module:
    """
    initialize a model with given architecture and state dictionary

    :param the_tabulatorium: table containing all relevant information
    :param architecture: location of the model architecture in the_tabulatorium[0]
    :param dataset: location of the dataset it was trained on in the_tabulatorium[1]
    :param version: location of the intermediate version in the tabulatorium[5][architecture][dataset]

    :return: the initialized neural network
    """

    model = the_tabulatorium[0][architecture][0]()
    model.load_state_dict(the_tabulatorium[4][architecture][dataset][version][0])

    return model


def update_checkpoint(
        the_tabulatorium: list,
        phase: str,
        location: list
) -> list:
    """
    given the task that was finished, generate the next tasks coordinates

    :param the_tabulatorium: table containing all relevant information
    :param phase: current phase (training, evaluating)
    :param location: coordinates of the finished task

    :return: list containing the new task:
        [0]: phase (string)
        [1]: location of the task (list)
    """

    if phase == 'training':
        architecture, dataset = location
        if dataset < len(the_tabulatorium[1]) - 1:
            checkpoint = ['training', [architecture, dataset + 1]]
        elif architecture < len(the_tabulatorium[0]) - 1:
            checkpoint = ['training', [architecture + 1, 0]]
        else:
            checkpoint = ['evaluating', [0, 0, 0, 0, 0]]
    elif phase == 'evaluating':
        architecture, dataset, explanation_method, metric, version = location
        if version < len(the_tabulatorium[4][architecture][dataset]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method, metric, version + 1]]
        elif metric < len(the_tabulatorium[3]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method, metric + 1, 0]]
        elif explanation_method < len(the_tabulatorium[2]) - 1:
            checkpoint = ['evaluating', [architecture, dataset, explanation_method + 1, 0, 0]]
        elif dataset < len(the_tabulatorium[1]) - 1:
            checkpoint = ['evaluation', [architecture, dataset + 1, 0, 0, 0]]
        elif architecture < len(the_tabulatorium[0]) - 1:
            checkpoint = ['evaluation', [architecture + 1, 0, 0, 0, 0]]
        else:
            checkpoint = ['done', []]
    else:
        checkpoint = ['done', []]

    return checkpoint
