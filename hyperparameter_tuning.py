import argparse
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from data_loader import ImageDataset
from dlc_practical_prologue import generate_pair_sets
from models import baseline_1, baseline_2, weight_sharing, weight_sharing_aux_loss
from train import train, train_with_auxiliary_loss


""" Contains model hyperparameter tuning methods, and can be run to tune hyperparameters for models """


def model_tuning(
        gen_model: Callable[[dict], Callable[[], Tuple[nn.Module, nn.Module, optim.Optimizer]]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, data.DataLoader, data.DataLoader, int],
                               Dict[str, List[float]]],
        num_epochs: int,
        rounds: int,
        hyperparameters: Dict[str, List],
        seed: Optional[int] = None,
        print_round_results: bool = True) -> dict:
    """
    Runs 5-fold cross validation to select the best learning rate, batch size and number of hidden units for a model

    :param gen_model: Function returning a function that generates the model to test. Takes as arguments a dictionary
        with parameters to the model.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the DataLoader containing the training data, the DataLoader containing the test data, and the
        number of epochs for which to train. Returns the loss and error rates on the training and test set after each
        epoch.
    :param num_epochs: The number of epochs to train the model for
    :param rounds: The number of rounds to do validation for
    :param hyperparameters: A dictionary mapping hyperparameter names to values to try for them. Can contain any
        parameter names that can be passed to the gen_model method, and 'batch_size'
    :param seed: The random seed if reproducibility is needed
    :param print_round_results: Whether to print intermediate results to the console
    :return: The best hyperparameter combination found
    """
    if seed:
        torch.manual_seed(seed)

    best_val_loss = 10000
    best_combination = None

    # Map hyperparameter values to list of lists of possible values
    possible_products = [
        [(hyperparameter_name, value) for value in possible_values]
        for hyperparameter_name, possible_values in hyperparameters.items()
    ]

    for hyperparameter_combination in product(*possible_products):
        hyperparameter_combination = dict(hyperparameter_combination)

        if print_round_results:
            print(f'    Testing with hyperparameters:\n'
                  f'      {hyperparameter_combination}')

        batch_size = hyperparameter_combination['batch_size']
        model_generator = gen_model(hyperparameter_combination)

        # Set round average validation loss
        average_val_loss = 0

        if seed:
            torch.manual_seed(seed)

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
            model, criterion, optimizer = model_generator()

            # Train the model
            results = train_method(model, optimizer, criterion, train_data, val_data, num_epochs)

            if results is None:
                print('      Skipping: loss became NaN')
                average_val_loss = None
                break

            final_val_loss = results['test_losses'][-1]
            average_val_loss += final_val_loss

            if print_round_results:
                print(f'        Round {i}: {final_val_loss:.4f}')

        if average_val_loss is not None:

            if print_round_results:
                print(f'    Average: {average_val_loss/rounds:.4f}')

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_combination = hyperparameter_combination

    return best_combination


def model_tuning_aux_loss(
        gen_model: Callable[[dict], Callable[[], Tuple[nn.Module, nn.Module, nn.Module, optim.Optimizer]]],
        train_method: Callable[[nn.Module, optim.Optimizer, nn.Module, nn.Module, data.DataLoader, data.DataLoader,
                                int, float], Dict[str, List[float]]],
        num_epochs: int,
        rounds: int,
        hyperparameters: Dict[str, List],
        seed: Optional[int] = None,
        print_round_results: bool = True) -> dict:
    """
    Runs 5-fold cross validation to select the best learning rate, batch size and number of hidden units for a model

    :param gen_model: Function returning a function that generates the model to test. Takes as arguments a dictionary
        with parameters to the model.
    :param train_method: The method used to train the model. Takes as input the model to train, the optimizer to use,
        the loss function, the auxiliary loss function, the DataLoader containing the training data, the DataLoader
        containing the test data, the number of epochs for which to train and the auxiliary loss weight. Returns the
        losses (full (sum of main and auxiliary), main and auxiliary) and error rates on the training and test set after
        each epoch.
    :param num_epochs: the number of epochs to train the model for
    :param rounds: the number of rounds to do validation for
    :param hyperparameters: A dictionary mapping hyperparameter names to values to try for them. Can contain any
        parameter names that can be passed to the gen_model method, 'aux_loss_weight' and 'batch_size'.
    :param seed: the random seed if reproducibility is needed
    :param print_round_results: whether to print intermediate results to the console
    :return: The batch size, learning rate, number of hidden units and aux. loss weight producing the best results
    """
    if seed:
        torch.manual_seed(seed)

    best_val_loss = 10000
    best_combination = None

    # Map hyperparameter values to list of lists of possible values
    possible_products = [
        [(hyperparameter_name, value) for value in possible_values]
        for hyperparameter_name, possible_values in hyperparameters.items()
    ]

    for hyperparameter_combination in product(*possible_products):
        hyperparameter_combination = dict(hyperparameter_combination)

        if print_round_results:
            print(f'    Testing with hyperparameters:\n'
                  f'      {hyperparameter_combination}')

        aux_weight = hyperparameter_combination['aux_loss_weight']
        batch_size = hyperparameter_combination['batch_size']
        model_generator = gen_model(hyperparameter_combination)

        # Set round average validation loss
        average_val_loss = 0

        if seed:
            torch.manual_seed(seed)

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
            model, criterion, aux_criterion, optimizer = model_generator()

            # Train the model
            results = train_method(model, optimizer, criterion, aux_criterion, train_data,
                                   val_data, num_epochs, aux_weight)

            if results is None:
                print('      Skipping: loss became NaN')
                average_val_loss = None
                break

            final_val_loss = results['test_main_losses'][-1]
            average_val_loss += final_val_loss

            if print_round_results:
                print(f'        Round {i}: {final_val_loss:.4f}')

        if average_val_loss is not None:

            if print_round_results:
                print(f'    Average: {average_val_loss / rounds:.4f}')

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_combination = hyperparameter_combination

    return best_combination


def tune_hyperparameters(models_to_tune):
    print(f'Tuning models {models_to_tune}')

    epochs = 25
    rounds = 10
    seed = 0

    batch_sizes = [10, 25, 50]
    lrs = [0.001, 0.01, 0.1]
    momentum = [0.1, 0.5, 0.9]
    weight_decay = [0.001, 0.01, 0.1]
    hidden_layer_units = [10, 25, 50, 100]

    if 'b1' in models_to_tune:
        print('Tuning Baseline 1:')
        base1_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'momentum': momentum,
            'weight_decay': weight_decay
        }
        base_1_combination = model_tuning(baseline_1, train, epochs, rounds, base1_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {base_1_combination}')

    if 'b2' in models_to_tune:
        print('Tuning Baseline 2:')
        base2_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'hidden_layer_units': hidden_layer_units,
        }
        base_2_combination = model_tuning(baseline_2, train, epochs, rounds, base2_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {base_2_combination}')

    if 'ws' in models_to_tune:
        print('Tuning WS:')
        ws_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'hidden_layer_units': hidden_layer_units,
        }
        ws_combination = model_tuning(weight_sharing, train, epochs, rounds, ws_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {ws_combination}')

    if 'wsal' in models_to_tune:
        print('Tuning WSAL:')
        wsal_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'hidden_layer_units': hidden_layer_units,
            'aux_loss_weight': [0.5, 1.0, 2.0, 5.0],
        }
        wsal_combination = model_tuning_aux_loss(weight_sharing_aux_loss, train_with_auxiliary_loss, epochs,
                                                 rounds, wsal_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {wsal_combination}')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Finds optimal hyperparameters for models')
    p.add_argument('--models', metavar='-m', nargs='*', default=['b1', 'b2', 'ws', 'wsal'],
                   help='Models on which to perform hyperparameter tuning. One or more of "b1", "b2", "ws", "wsal"')
    args = p.parse_args()

    tune_hyperparameters(args.models)
