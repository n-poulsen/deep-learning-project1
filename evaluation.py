from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from dlc_practical_prologue import generate_pair_sets
from data_loader import train_loader, test_loader

from data_loader import ImageDataset

""" Contains model evaluation methods """


def model_tuning(
        gen_model: Callable[[float], Tuple[nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, data.DataLoader, data.DataLoader, int],
                               Tuple[List[float], List[float], List[float], List[float]]],
        num_epochs: int,
        batch_sizes: List[int],
        learning_rates: List[float],
        seed: Optional[int] = None,
        print_round_results: bool = True) -> Tuple[int, float]:
    """
    Runs 5-fold cross validation to select the best learning rate and batch size for a model

    :param gen_model: Function generating the model to test. Takes as arguments the learning rate for the optimizer.
        Returns the model, loss function and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data, and the
        number of epochs for which to train. Returns the loss and error rates on the training and test set after each
        epoch.
    :param num_epochs:
    :param batch_sizes:
    :param learning_rates:
    :param seed:
    :param print_round_results:
    """
    if seed:
        torch.manual_seed(seed)

    # Load data
    train_x, train_y, train_c, _, _, _ = generate_pair_sets(1000)
    image_dataset = ImageDataset(train_x, train_y, train_c)
    folds = data.random_split(image_dataset, [200, 200, 200, 200, 200], generator=torch.Generator().manual_seed(0))

    best_val_loss = 10000
    best_batch_size = None
    best_learning_rate = None

    for batch_size in batch_sizes:
        for lr in learning_rates:

            if seed:
                torch.manual_seed(seed)

            average_val_loss = 0

            if print_round_results:
                print(f'Testing batch_size {batch_size}, lr={lr}')

            # Run 5-fold cross validation on the hyperparameters
            for fold_left_out in range(5):
                # Create the training and validation data loaders for the folds
                train_folds = folds[:fold_left_out] + folds[fold_left_out + 1:]
                train_dataset = data.ConcatDataset(train_folds)
                val_dataset = folds[fold_left_out]
                train_data = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_data = data.DataLoader(val_dataset, batch_size=1, shuffle=False)

                # Generate the model, criterion and optimizer
                model, criterion, optimizer = gen_model(lr)

                # Train the model
                _, _, val_loss, _ = train_method(model, optimizer, criterion, train_data, val_data, num_epochs)
                final_val_loss = val_loss[-1]
                average_val_loss += final_val_loss

                if print_round_results:
                    print(f'    Round {fold_left_out}: {final_val_loss:.4f}')

            if print_round_results:
                print(f'  Average: {average_val_loss/5:.4f}')

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_batch_size = batch_size
                best_learning_rate = lr

    return best_batch_size, best_learning_rate


def model_eval(
        gen_model: Callable[[], Tuple[nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, data.DataLoader, data.DataLoader, int],
                               Tuple[List[float], List[float], List[float], List[float]]],
        rounds: int,
        num_epochs: int,
        batch_size: int,
        seed: Optional[int] = None,
        print_round_results: bool = True) -> List[Tuple[List[float], List[float], List[float], List[float]]]:
    """
    Computes the performance of a model over a given number of rounds, computing the average loss and error rate over
    the test set for each one.

    This method prints the results for each round to the console.

    :param gen_model: Function generating the model to test. Returns the model, loss function and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data, and the
        number of epochs for which to train. Returns the loss and error rates on the training and test set after each
        epoch.
    :param rounds: The number of rounds used to evaluate the performance of the model
    :param num_epochs: The number of epochs for which to train.
    :param batch_size: The mini-batch size to use for training.
    :param seed: Random seed to use for reproducibility
    :param print_round_results: Whether to print the results for each round while evaluating models
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
        tr_loss, tr_err, te_loss, te_err = train_method(model, optimizer, criterion, train_data, test_data, num_epochs)
        round_results.append((tr_loss, tr_err, te_loss, te_err))

        # Print the results for the round
        if print_round_results:
            print(f'Round {test_round + 1}')
            print(f'Test results:')
            print(f'  Train Loss:       {tr_loss[-1]:.3f}')
            print(f'  Train Error Rate:  {100 * tr_err[-1]:.2f}%')
            print(f'  Test Error Rate:   {100 * te_err[-1]:.2f}%')
            print("-" * 50, "\n")

    return round_results


def aux_model_eval(
        gen_model: Callable[[], Tuple[nn.Module, nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, nn.Module, data.DataLoader, data.DataLoader, int],
                               Tuple[List[float], List[float], List[float], List[float]]],
        rounds: int,
        num_epochs: int,
        seed: Optional[int] = None,
        print_round_results: bool = True) -> List[Tuple[List[float], List[float], List[float], List[float]]]:
    """
    Computes the performance of a model containing an auxiliary loss over a given number of rounds, computing the
    average loss and error rate over the test set for each one.

    This method prints the results for each round to the console.

    :param gen_model: Function generating the model to test. Returns the model, loss function, auxiliary loss function
        and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the auxiliary loss function, the DataLoader containing the training data, the DataLoader
        containing the test data, and the number of epochs for which to train. Returns the loss and error rates on the
        training and test set after each epoch.
    :param rounds: The number of rounds used to evaluate the performance of the model
    :param num_epochs: The number of epochs for which to train.
    :param seed: Random seed to use for reproducibility
    :param print_round_results: Whether to print the results for each round while evaluating models
    :return: For each round, the training loss, training error, test loss and test error for each epoch.
    """
    if seed:
        torch.manual_seed(seed)

    round_results = []

    for test_round in range(rounds):
        # Load data
        train_x, train_y, train_c, test_x, test_y, test_c = generate_pair_sets(1000)
        train_data = train_loader(train_x, train_y, train_c, 10)
        test_data = test_loader(test_x, test_y, test_c)

        # Generate the model, criterion, auxiliary criterion and optimizer
        model, crit, aux_crit, opti = gen_model()

        # Train the model
        tr_loss, tr_err, te_loss, te_err = train_method(model, opti, crit, aux_crit, train_data, test_data, num_epochs)
        round_results.append((tr_loss, tr_err, te_loss, te_err))

        # Print the results for the round
        if print_round_results:
            print(f'Round {test_round + 1}')
            print(f'Test results:')
            print(f'  Train Loss:       {tr_loss[-1]:.3f}')
            print(f'  Train Error Rate:  {100 * tr_err[-1]:.2f}%')
            print(f'  Test Error Rate:   {100 * te_err[-1]:.2f}%')
            print("-" * 50, "\n")

    return round_results


def model_perf(per_round_error_rate: List[float]):
    """
    Computes the mean and standard deviation of the error rate for a model, and plots the error rate for each round.

    :param per_round_error_rate: The error rate over each round of the training
    :return: None
    """
    error_rate_avg = torch.tensor(per_round_error_rate).mean().item()
    error_rate_std = torch.tensor(per_round_error_rate).std().item()
    print(f"Error Rate Average: {error_rate_avg:.3f}%;  Test error standard deviations: {error_rate_std:.3f}%")

    plt.figure(figsize=(15, 3))
    plt.title('Model Performance')
    plt.xlabel('ROUND')
    plt.ylabel('Error rate')
    plt.plot(range(1, len(per_round_error_rate) + 1), per_round_error_rate)
    plt.show()
