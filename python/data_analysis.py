import matplotlib.pyplot as plt


def compare_over_training(metrics: list, data: dict) -> None:
    steps_accuracy = [data[model_version][1] for model_version in data]
    for metric in metrics:
        results = [data[model_version][2][metric] for model_version in data]
        plt.plot(steps_accuracy, results)
    plt.show()
