from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dlc_practical_prologue import generate_pair_sets
from data_loader import train_loader, test_loader


""" Contains model evaluation methods """


def model_evaluation(
        gen_model: Callable[[], Tuple[nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, data.DataLoader, data.DataLoader, int],
                               Dict[str, List[float]]],
        rounds: int,
        num_epochs: int,
        batch_size: int,
        seed: Optional[int] = None,
        log_results: bool = True) -> List[Tuple[List[float], List[float], List[float], List[float]]]:
    """
    Computes the performance of a model over a given number of rounds, computing the average loss and error rate over
    the test set for each one.

    This method prints the results for each round to the console.

    :param gen_model: Function generating the model to test. Returns the model, loss function and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data and the
        number of epochs for which to train. Returns the loss and error rates on the training and test set after each
        epoch.
    :param rounds: The number of rounds used to evaluate the performance of the model
    :param num_epochs: The number of epochs for which to train.
    :param batch_size: The mini-batch size to use for training.
    :param seed: Random seed to use for reproducibility
    :param log_results: Whether to print the results for each round while evaluating models
    :return: For each round, the training loss, training error, test loss and test error for each epoch.
    """
    if seed:
        torch.manual_seed(seed)

    round_results = []

    for test_round in range(rounds):
        # Load data
        train_x, train_y, train_c, test_x, test_y, test_c = generate_pair_sets(1000)
        train_data = train_loader(train_x, train_y, train_c, batch_size)
        test_data = test_loader(test_x, test_y, test_c)

        # Generate the model, criterion and optimizer
        model, criterion, optimizer = gen_model()

        # Train the model
        results = train_method(model, optimizer, criterion, train_data, test_data, num_epochs)

        if results is None:
            raise OverflowError('Loss became NaN in training. Try different hyperparameters')

        round_results.append((
            results['train_losses'],
            results['train_error_rates'],
            results['test_losses'],
            results['test_error_rates'],
        ))

        # Print the results for the round
        if log_results:
            print(f'    Round {test_round + 1}')
            print(f'    Test results:')
            print(f'      Train Loss:       {results["train_losses"][-1]:.3f}')
            print(f'      Train Error Rate:  {100 * results["train_error_rates"][-1]:.2f}%')
            print(f'      Test Error Rate:   {100 * results["test_error_rates"][-1]:.2f}%')
            print(f'    {"-" * 50}\n')

    return round_results


def model_evaluation_with_auxiliary_loss(
        gen_model: Callable[[], Tuple[nn.Module, nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, nn.Module, data.DataLoader, data.DataLoader,
                                int, float], Dict[str, List[float]]],
        auxiliary_loss_weight: float,
        rounds: int,
        num_epochs: int,
        batch_size: int,
        seed: Optional[int] = None,
        log_results: bool = True) -> List[Tuple[List[float], List[float], List[float], List[float]]]:
    """
    Computes the performance of a model containing an auxiliary loss over a given number of rounds, computing the
    average loss and error rate over the test set for each one.

    This method prints the results for each round to the console.

    :param gen_model: Function generating the model to test. Returns the model, loss function, auxiliary loss function
        and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the auxiliary loss function, the DataLoader containing the training data, the DataLoader
        containing the test data, the number of epochs for which to train and the weight to apply to the auxiliary loss.
        Returns the loss and error rates on the training and test set after each epoch.
    :param auxiliary_loss_weight: The weight to apply to the auxiliary loss
    :param rounds: The number of rounds used to evaluate the performance of the model
    :param num_epochs: The number of epochs for which to train.
    :param batch_size: The mini-batch size to use for training.
    :param seed: Random seed to use for reproducibility
    :param log_results: Whether to print the results for each round while evaluating models
    :return: For each round, the training loss, training error, test loss and test error for each epoch.
    """
    if seed:
        torch.manual_seed(seed)

    round_results = []

    for test_round in range(rounds):
        # Load data
        train_x, train_y, train_c, test_x, test_y, test_c = generate_pair_sets(1000)
        train_data = train_loader(train_x, train_y, train_c, batch_size)
        test_data = test_loader(test_x, test_y, test_c)

        # Generate the model, criterion, auxiliary criterion and optimizer
        model, criterion, aux_criterion, optimizer = gen_model()

        # Train the model
        results = train_method(model, optimizer, criterion, aux_criterion, train_data, test_data, num_epochs,
                               auxiliary_loss_weight)

        if results is None:
            raise OverflowError('Loss became NaN in training. Try different hyperparameters')

        # Process results
        round_results.append((
            results['train_losses'],
            results['train_error_rates'],
            results['test_losses'],
            results['test_error_rates'],
        ))

        # Print the results for the round
        if log_results:
            print(f'    Round {test_round + 1}')
            print(f'    Test results:')
            print(f'      Train Loss:       {results["train_losses"][-1]:.3f}')
            print(f'      Train Error Rate:  {100 * results["train_error_rates"][-1]:.2f}%')
            print(f'      Test Error Rate:   {100 * results["test_error_rates"][-1]:.2f}%')
            print(f'    {"-" * 50}\n')

    return round_results
