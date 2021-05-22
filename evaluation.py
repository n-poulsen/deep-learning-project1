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
        gen_model: Callable[[dict], Tuple[nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, data.DataLoader, data.DataLoader, int],
                               Tuple[List[float], List[float], List[float], List[float]]],
        num_epochs: int,
        rounds: int,
        batch_sizes: List[int],
        learning_rates: List[float],
        hidden_layer_units: List[int],
        seed: Optional[int] = None,
        print_round_results: bool = True) -> Tuple[int, float, int]:
    """
    Runs 5-fold cross validation to select the best learning rate, batch size and number of hidden units for a model

    :param gen_model: Function generating the model to test. Takes as arguments a dictionary with the keys 'lr', setting
        the learning rate for the optimizer, and 'hidden_units', setting the number of hidden units in the model.
        Returns the model, loss function and optimizer to test.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data, and the
        number of epochs for which to train. Returns the loss and error rates on the training and test set after each
        epoch.
    :param num_epochs: the number of epochs to train the model for
    :param rounds: the number of rounds to do validation for
    :param batch_sizes: the batch sizes to try
    :param learning_rates: the learning rates to try
    :param hidden_layer_units: the number of hidden layer units to try
    :param seed: the random seed if reproducibility is needed
    :param print_round_results: whether to print intermediate results to the console
    :return: The batch size, learning rate and number of hidden units producing the best results
    """
    if seed:
        torch.manual_seed(seed)

    best_val_loss = 10000
    best_batch_size = None
    best_learning_rate = None
    best_hidden_units = None

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for hidden_units in hidden_layer_units:
                # Set round average validation loss
                average_val_loss = 0

                if seed:
                    torch.manual_seed(seed)

                if print_round_results:
                    print(f'Testing batch_size {batch_size}, lr={lr}, units={hidden_units}')

                for i in range(rounds):
                    # Load data
                    train_x, train_y, train_c, _, _, _ = generate_pair_sets(1000)
                    image_dataset = ImageDataset(train_x, train_y, train_c)

                    # Split into train and validation sets
                    train_set, val_set = data.random_split(image_dataset, [800, 200],
                                                           generator=torch.Generator().manual_seed(0))
                    train_data = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
                    val_data = data.DataLoader(val_set, batch_size=1, shuffle=False)

                    # Generate the model, criterion and optimizer
                    model, criterion, optimizer = gen_model({
                        'lr': lr,
                        'hidden_units': hidden_units,
                    })

                    # Train the model
                    _, _, val_loss, _ = train_method(model, optimizer, criterion, train_data, val_data, num_epochs)
                    final_val_loss = val_loss[-1]
                    average_val_loss += final_val_loss

                    if print_round_results:
                        print(f'    Round {i}: {final_val_loss:.4f}')

                if print_round_results:
                    print(f'  Average: {average_val_loss/rounds:.4f}')

                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    best_batch_size = batch_size
                    best_learning_rate = lr
                    best_hidden_units = hidden_units

    return best_batch_size, best_learning_rate, best_hidden_units


def model_evaluation(
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
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data and the
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
            print(f'    Round {test_round + 1}')
            print(f'    Test results:')
            print(f'      Train Loss:       {tr_loss[-1]:.3f}')
            print(f'      Train Error Rate:  {100 * tr_err[-1]:.2f}%')
            print(f'      Test Error Rate:   {100 * te_err[-1]:.2f}%')
            print(f'    {"-" * 50}\n')

    return round_results


def model_evaluation_with_auxiliary_loss(
        gen_model: Callable[[], Tuple[nn.Module, nn.Module, nn.Module, optim.Optimizer]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, nn.Module, data.DataLoader, data.DataLoader,
                                int, float], Tuple[List[float], List[float], List[float], List[float]]],
        auxiliary_loss_weight: float,
        rounds: int,
        num_epochs: int,
        batch_size: int,
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
        containing the test data, the number of epochs for which to train and the weight to apply to the auxiliary loss.
        Returns the loss and error rates on the training and test set after each epoch.
    :param auxiliary_loss_weight: The weight to apply to the auxiliary loss
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

        # Generate the model, criterion, auxiliary criterion and optimizer
        model, criterion, aux_criterion, optimizer = gen_model()

        # Train the model
        tr_loss, tr_err, te_loss, te_err = train_method(
            model, optimizer, criterion, aux_criterion, train_data, test_data, num_epochs, auxiliary_loss_weight)

        # Process results
        round_results.append((tr_loss, tr_err, te_loss, te_err))

        # Print the results for the round
        if print_round_results:
            print(f'    Round {test_round + 1}')
            print(f'    Test results:')
            print(f'      Train Loss:       {tr_loss[-1]:.3f}')
            print(f'      Train Error Rate:  {100 * tr_err[-1]:.2f}%')
            print(f'      Test Error Rate:   {100 * te_err[-1]:.2f}%')
            print(f'    {"-" * 50}\n')

    return round_results


def parse_results(round_results: List[Tuple[List[float], List[float], List[float], List[float]]]):
    """


    :param round_results:
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


def model_performance(per_round_error_rate: List[float]):
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
