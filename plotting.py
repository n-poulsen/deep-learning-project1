from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from helpers import compute_statistics


def plot_model_training_loss_per_epoch(mean_train_loss: List[float], model_name: str):
    """
      Plot the mean train loss per epoch.

      :param mean_train_loss: The mean training loss per epoch
      :param model_name: the name of the model
      :return: None
    """
    plt.figure(figsize=(15, 5))
    plt.title(f'{model_name} model Training Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.grid()
    plt.plot(range(1, len(mean_train_loss) + 1), mean_train_loss)
    plt.show()


def plot_model_error_rate_per_epoch(mean_train_error: List[float], std_train_error: List[float],
                                    mean_test_error: List[float], std_test_error: List[float], model_name: str):
    """
      Plot the mean error rate and std per epoch.

      :param mean_train_error: The mean training error rate per epoch
      :param std_train_error: The training error std per epoch
      :param mean_test_error: The mean test error rate per epoch
      :param std_test_error: The test error std per epoch
      :param model_name: the name of the model
      :return: None
    """
    plt.figure(figsize=(15, 5))
    plt.title(f'{model_name} model Error Rate')
    plt.xlabel('EPOCH')
    plt.ylabel('ERROR RATE (%)')
    plt.grid()
    plt.plot(range(1, len(mean_test_error) + 1), mean_test_error, label='Mean Test Error')
    plt.fill_between(range(1, len(mean_test_error) + 1), mean_test_error - std_test_error,
                     mean_test_error + std_test_error, alpha=.1, label='Mean Test Error std')
    plt.plot(range(1, len(mean_train_error) + 1), mean_train_error, label='Mean Train Error')
    plt.fill_between(range(1, len(mean_train_error) + 1), mean_train_error - std_train_error,
                     mean_train_error + std_train_error, alpha=.1, label='Mean Train std')
    plt.legend()
    plt.show()


def error_rate_box_plots(model_test_errors: Dict[str, List[List[float]]]):
    """
    Computes the mean and standard deviation of the error rate for a model, and plots the error rate for each round.

    :param model_test_errors: The error rate over each round of the training
    :return: None
    """
    model_final_test_errors = {}
    for model_name, round_test_errors in model_test_errors.items():

        per_round_test_errors = []
        for test_error in round_test_errors:
            per_round_test_errors.append(test_error[-1])

        model_final_test_errors[model_name] = per_round_test_errors

    labels = [m for m in model_final_test_errors.keys()]
    test_errors = [te for te in model_final_test_errors.values()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)

    ax.set_xlabel('Model', labelpad=25, fontsize=32)
    ax.set_ylabel('Error Rate', labelpad=25, fontsize=32)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)

    ax.boxplot(test_errors, labels=labels)

    plt.tight_layout()
    plt.show()


def plot_results(round_results: List[Tuple[List[float], List[float], List[float], List[float]]], model_name: str):
    """
        Compute the mean train loss, mean train loss std, mean train error, mean train error std, mean test error
        and mean test error std. Each one aggregated per epoch. Then plot the mean train loss per epoch and the mean
        and standard deviation of the error rate for a model, and plots the error rate for each round.

        :param round_results: results of training a model during a number of rounds, where the results for each round
        are a tuple containing the training loss, training error rate, test loss and test error rate per epoch
        :param model_name: the name of the model
        :return: None
    """

    mean_train_loss, std_train_loss, mean_train_error, std_train_error, mean_test_error, std_test_error = compute_statistics(
        round_results)

    plot_model_training_loss_per_epoch(mean_train_loss, model_name)
    plot_model_error_rate_per_epoch(mean_train_error, std_train_error, mean_test_error, std_test_error, model_name)
