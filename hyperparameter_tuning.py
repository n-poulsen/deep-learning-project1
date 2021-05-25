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


def load_train_and_validation_data(batch_size: int) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Generates a training set containing 1000 pairs from MNIST and splits it into a training set containing 800 samples
    and a validation set containing 200 pairs.

    :param batch_size: the batch size for the training dataloader.
    :return: a training and a validation dataloader
    """
    # Load data
    train_x, train_y, train_c, _, _, _ = generate_pair_sets(1000)
    image_dataset = ImageDataset(train_x, train_y, train_c)

    # Split into train and validation sets
    train_set, val_set = data.random_split(image_dataset, [800, 200], generator=torch.Generator().manual_seed(0))
    train_data = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_data = data.DataLoader(val_set, batch_size=1, shuffle=False)
    return train_data, val_data


def model_tuning(
        gen_model: Callable[[dict], Callable[[], Tuple[nn.Module, nn.Module, optim.Optimizer]]],
        num_epochs: int,
        rounds: int,
        hyperparameters: Dict[str, List],
        seed: Optional[int] = None,
        log_results: bool = True) -> dict:
    """
    Runs cross validation to select the best learning rate, batch size and number of hidden units for a model.
    Repeatedly generates a training set and splits into two chunk. The first chunk, 80% of the data, is used to train
    the model. The second chunk, 20% of the data, is used to test it.

    For each combination of hyperparameters, trains {rounds} models. Returns the hyperparameters leading to the best
    validation loss. If a model is testing at more than 150% of the current best validation loss halfway through the
    rounds, doesn't train for the second half as the hyperparameters are very unlikely to perform better.

    If the hyperparameters lead to the model diverging (e.g., due to a learning rate that is too large), skips training
    the other models with the same hyperparameters.

    :param gen_model: Function returning a function that generates the model to test. Takes as arguments a dictionary
        with parameters to the model.
    :param num_epochs: The number of epochs to train the model for
    :param rounds: The number of rounds to do validation for
    :param hyperparameters: A dictionary mapping hyperparameter names to values to try for them. Can contain any
        parameter names that can be passed to the gen_model method, and 'batch_size'
    :param seed: The random seed if reproducibility is needed
    :param log_results: Whether to print intermediate results to the console
    :return: The best hyperparameters for the model
    """
    best_val_loss = 10000
    best_combination = None

    # Map hyperparameter values to list of lists of possible values
    possible_products = [
        [(hyperparameter_name, value) for value in possible_values]
        for hyperparameter_name, possible_values in hyperparameters.items()
    ]

    for hyperparameter_combination in product(*possible_products):
        hyperparameter_combination = dict(hyperparameter_combination)

        if log_results:
            print(f'    Testing with hyperparameters:\n'
                  f'      {hyperparameter_combination}')

        batch_size = hyperparameter_combination['batch_size']
        model_generator = gen_model(hyperparameter_combination)

        # Set round average validation loss
        average_val_loss = 0

        if seed is not None:
            torch.manual_seed(seed)

        for i in range(rounds):
            # Generate the training and validation data loaders, the model, criterion and optimizer
            train_data, val_data = load_train_and_validation_data(batch_size)
            model, criterion, optimizer = model_generator()
            # Train the model
            results = train(model, optimizer, criterion, train_data, val_data, num_epochs)

            if results is None:
                print('      Skipping: loss became NaN')
                average_val_loss = None
                break

            final_val_loss = results['test_losses'][-1]
            average_val_loss += final_val_loss

            if log_results:
                print(f'        Round {i}: {final_val_loss:.4f}')

            if i + 1 == rounds // 2:
                halfway_average_loss = average_val_loss/i
                if halfway_average_loss > 1.5 * best_val_loss:
                    print('      Skipping: average loss too high halfway through rounds')
                    # Compensate for the fact that we only went through half the rounds
                    average_val_loss = 2 * average_val_loss
                    break

        if average_val_loss is not None:
            average_val_loss = average_val_loss / rounds

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_combination = hyperparameter_combination

            if log_results:
                print(f'    Average: {average_val_loss:.4f}')

    return best_combination


def model_tuning_aux_loss(
        gen_model: Callable[[dict], Callable[[], Tuple[nn.Module, nn.Module, nn.Module, optim.Optimizer]]],
        num_epochs: int,
        rounds: int,
        hyperparameters: Dict[str, List],
        seed: Optional[int] = None,
        log_results: bool = True) -> dict:
    """
    Runs cross validation to select the best learning rate, batch size and number of hidden units for a model.
    Repeatedly generates a training set and splits into two chunk. The first chunk, 80% of the data, is used to train
    the model. The second chunk, 20% of the data, is used to test it.

    For each combination of hyperparameters, trains {rounds} models. Returns the hyperparameters leading to the best
    main validation loss (the loss without the auxiliary loss). If a model is testing at more than 150% of the current
    best validation loss halfway through the rounds, doesn't train for the second half as the hyperparameters are very
    unlikely to perform better.

    If the hyperparameters lead to the model diverging (e.g., due to a learning rate that is too large), skips training
    the other models with the same hyperparameters.

    :param gen_model: Function returning a function that generates the model to test. Takes as arguments a dictionary
        with parameters to the model.
    :param num_epochs: the number of epochs to train the model for
    :param rounds: the number of rounds to do validation for
    :param hyperparameters: A dictionary mapping hyperparameter names to values to try for them. Can contain any
        parameter names that can be passed to the gen_model method, 'aux_loss_weight' and 'batch_size'.
    :param seed: the random seed if reproducibility is needed
    :param log_results: whether to print intermediate results to the console
    :return: The best hyperparameters for the model
    """
    best_val_loss = 10000
    best_combination = None

    # Map hyperparameter values to list of lists of possible values
    possible_products = [
        [(hyperparameter_name, value) for value in possible_values]
        for hyperparameter_name, possible_values in hyperparameters.items()
    ]

    for hyperparameter_combination in product(*possible_products):
        hyperparameter_combination = dict(hyperparameter_combination)

        if log_results:
            print(f'    Testing with hyperparameters:\n'
                  f'      {hyperparameter_combination}')

        aux_weight = hyperparameter_combination['aux_loss_weight']
        batch_size = hyperparameter_combination['batch_size']
        model_generator = gen_model(hyperparameter_combination)

        # Set round average validation loss
        average_val_loss = 0

        if seed is not None:
            torch.manual_seed(seed)

        for i in range(rounds):
            # Generate the training and validation data loaders, the model, criterions and optimizer
            train_data, val_data = load_train_and_validation_data(batch_size)
            model, criterion, aux_criterion, optimizer = model_generator()
            # Train the model
            results = train_with_auxiliary_loss(
                model, optimizer, criterion, aux_criterion, train_data, val_data, num_epochs, aux_weight)

            if results is None:
                print('      Skipping: loss became NaN')
                average_val_loss = None
                break

            final_val_loss = results['test_main_losses'][-1]
            average_val_loss += final_val_loss

            if log_results:
                print(f'        Round {i}: {final_val_loss:.4f}')

            if i + 1 == rounds // 2:
                halfway_average_loss = average_val_loss/i
                if halfway_average_loss > 1.5 * best_val_loss:
                    print('      Skipping: average loss too high halfway through rounds')
                    # Compensate for the fact that we only went through half the rounds
                    average_val_loss = 2 * average_val_loss
                    break

        if average_val_loss is not None:
            average_val_loss = average_val_loss / rounds

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_combination = hyperparameter_combination

            if log_results:
                print(f'    Average: {average_val_loss:.4f}')

    return best_combination


def tune_hyperparameters(models_to_tune):
    print(f'Tuning models {models_to_tune}')

    epochs = 25
    rounds = 10
    seed = 0

    batch_sizes = [25]
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    weight_decay = [0, 0.001, 0.01, 0.1]
    hidden_layer_units = [10, 25, 50]

    if 'b1' in models_to_tune:
        print('Tuning Baseline 1:')
        base1_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'weight_decay': weight_decay
        }
        base_1_combination = model_tuning(baseline_1, epochs, rounds, base1_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {base_1_combination}')

    if 'b2' in models_to_tune:
        print('Tuning Baseline 2:')
        base2_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'weight_decay': weight_decay,
            'hidden_layer_units': hidden_layer_units,
        }
        base_2_combination = model_tuning(baseline_2, epochs, rounds, base2_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {base_2_combination}')

    if 'ws' in models_to_tune:
        print('Tuning WS:')
        ws_hyperparameters = {
            'batch_size': batch_sizes,
            'lr': lrs,
            'weight_decay': weight_decay,
            'hidden_layer_units': hidden_layer_units,
        }
        ws_combination = model_tuning(weight_sharing, epochs, rounds, ws_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {ws_combination}')

    if 'wsal' in models_to_tune:
        print('Tuning WSAL:')
        wsal_hyperparameters = {
            'batch_size': batch_sizes,
            'hidden_layer_units': hidden_layer_units,
            'lr': lrs,
            'weight_decay': weight_decay,
            'aux_loss_weight': [0.5, 1.0, 2.0, 5.0, 10.0],
        }
        print('  Testing Values:')
        for k, v in wsal_hyperparameters.items():
            print(f'    {k}: {v}')
        wsal_combination = model_tuning_aux_loss(
            weight_sharing_aux_loss, epochs, rounds, wsal_hyperparameters, seed=seed)
        print(f'  -> Best combination found: {wsal_combination}')


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Finds optimal hyperparameters for models')
    p.add_argument('--models', metavar='-m', nargs='*', default=['b1', 'b2', 'ws', 'wsal'],
                   help='Models on which to perform hyperparameter tuning. One or more of "b1", "b2", "ws", "wsal"')
    args = p.parse_args()

    tune_hyperparameters(args.models)
