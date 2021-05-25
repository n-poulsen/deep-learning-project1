from typing import Dict, List, Tuple, Callable

import matplotlib.pyplot as plt
import torch


def print_divider():
    """ Prints a divider consisting of dashes to the console """
    print()
    print("-" * 80)
    print("-" * 80)
    print("-" * 80)
    print()


def log_model_information(model_generating_function: Callable, hyperparameters: dict):
    """
    Prints the number of trainable parameters in a model, as well as it's hyperparameters, to the console.

    :param model_generating_function: the function taking the hyperparameters as a argument and returning the model
        generating function.
    :param hyperparameters:
    :return:
    """
    model = model_generating_function(hyperparameters)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable Parameters: {num_params}')
    for parameter, value in hyperparameters.items():
        print(f'  {parameter}: {value}')
    print()


def parse_results(round_results: List[Tuple[List[float], List[float], List[float], List[float]]]):
    """
    Compute the mean and standard deviation of a model trained for some number of rounds, and prints the results to the
    console.

    :param round_results: results of training a model during a number of rounds, where the results for each round are a
        tuple containing the training loss, training error rate, test loss and test error rate per epoch
    :return: None
    """
    train_loss = []
    train_error = []
    test_error = []

    for tr_loss, tr_err, _, te_err in round_results:
        train_loss.append(tr_loss[-1])
        train_error.append(tr_err[-1])
        test_error.append(te_err[-1])

    train_loss = torch.tensor(train_loss)
    train_error = torch.tensor(train_error)
    test_error = torch.tensor(test_error)

    #  Mean of the values per epoch
    mean_train_loss = train_loss.mean()
    mean_train_error = train_error.mean()
    mean_test_error = test_error.mean()

    #  Standard deviation of the values per epoch
    std_train_loss = train_loss.std()
    std_train_error = train_error.std()
    std_test_error = test_error.std()

    print(f'Results:')
    print(f'    Mean Training Loss:  {mean_train_loss:.2f}')
    print(f'    Mean Training Error: {100 * mean_train_error:.2f}%')
    print(f'    Mean Testing Error:  {100 * mean_test_error:.2f}%')
    print()
    print(f'    STD of Training Loss:   {std_train_loss:.4f}')
    print(f'    STD of Training Error:  {100 * std_train_error:.2f}')
    print(f'    STD of Testing Error:   {100 * std_test_error:.2f}')


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
