from typing import List, Tuple, Callable

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

    # If only one round is ran, we only display the training loss, training error and testing error for that round
    if len(round_results) <2:
        print(f'Results:')
        print(f'Training Loss:  {mean_train_loss:.2f}')
        print(f'Training Error: {100 * mean_train_error:.2f}%')
        print(f'Testing Error:  {100 * mean_test_error:.2f}%')
    # If more than one round is ran, we  display the mean and standard deviation per epoch of the training losses,
    # training errors and testing errors
    else:
        print(f'Results:')
        print(f'    Mean Training Loss:  {mean_train_loss:.2f}')
        print(f'    Mean Training Error: {100 * mean_train_error:.2f}%')
        print(f'    Mean Testing Error:  {100 * mean_test_error:.2f}%')
        print()
        print(f'    STD of Training Loss:   {std_train_loss:.4f}')
        print(f'    STD of Training Error:  {100 * std_train_error:.2f}')
        print(f'    STD of Testing Error:   {100 * std_test_error:.2f}')


def compute_statistics(round_results: List[Tuple[List[float], List[float], List[float], List[float]]]):
    """
       Compute the mean and standard deviation per epoch of a model trained for some number of rounds

       :param round_results: results of training a model during a number of rounds, where the results for each round are a
           tuple containing the training loss, training error rate, test loss and test error rate per epoch

       :return: The mean train loss, mean train loss std, mean train error, mean train error std, mean test error and mean test error std.
           Each one aggregated per epoch
    """

    train_loss = []
    train_error = []
    test_error = []

    for tr_loss, tr_err, _, te_err in round_results:
        train_loss.append(tr_loss)
        train_error.append(tr_err)
        test_error.append(te_err)

    train_loss = torch.tensor(train_loss)
    train_error = torch.tensor(train_error)
    test_error = torch.tensor(test_error)

    # Mean of the values per epoch
    mean_train_loss = train_loss.mean(dim=0)
    mean_train_error = train_error.mean(dim=0)
    mean_test_error = test_error.mean(dim=0)

    # Standard deviation of the values per epoch
    std_train_loss = train_loss.std(dim=0)
    std_train_error = train_error.std(dim=0)
    std_test_error = test_error.std(dim=0)

    return mean_train_loss, std_train_loss, mean_train_error, std_train_error, mean_test_error, std_test_error
