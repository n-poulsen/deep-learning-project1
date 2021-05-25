import argparse

from models import baseline_1, baseline_2, weight_sharing, weight_sharing_aux_loss
from helpers import error_rate_box_plots, parse_results, print_divider, log_model_information
from evaluation import model_evaluation, model_evaluation_with_auxiliary_loss
from train import train, train_with_auxiliary_loss


# Hyperparameters used to train the baseline_1 model
baseline_1_parameters = {
    'batch_size': 25,
    'lr': 1e-4,
    'weight_decay': 0.1
}


# Hyperparameters used to train the baseline_2 model
baseline_2_parameters = {
    'batch_size': 25,
    'lr': 1e-4,
    'weight_decay': 0.0,
    'hidden_layer_units': 50,
}


# Hyperparameters used to train the weight sharing model
ws_parameters = {
    'batch_size': 25,
    'lr': 1e-4,
    'weight_decay': 0.0,
    'hidden_layer_units': 50,
}


# Hyperparameters used to train the weight sharing with auxiliary loss model
wsal_parameters = {
    'batch_size': 25,
    'lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 0.0,
    'hidden_layer_units': 50,
    'aux_loss_weight': 5.0,
}


def evaluate_models(rounds: int, log_results: bool):
    epochs = 25
    seed = 0

    print(f'\n\nEvaluating models through {rounds} rounds of training.')
    print(f'Models trained for {epochs} epochs.')
    print_divider()

    print('Evaluating Baseline 1 Model')
    log_model_information(baseline_1, baseline_1_parameters)
    baseline_1_results = model_evaluation(baseline_1(baseline_1_parameters), train, rounds, epochs,
                                          baseline_1_parameters['batch_size'], seed=seed, log_results=log_results)
    parse_results(baseline_1_results)
    print_divider()

    print('Evaluating Baseline 2 Model')
    log_model_information(baseline_2, baseline_2_parameters)
    baseline_2_results = model_evaluation(baseline_2(baseline_2_parameters), train, rounds, epochs,
                                          baseline_2_parameters['batch_size'], seed=seed, log_results=log_results)
    parse_results(baseline_2_results)
    print_divider()

    print('Evaluating Weight Sharing Model')
    log_model_information(weight_sharing, ws_parameters)
    ws_results = model_evaluation(weight_sharing(ws_parameters), train, rounds, epochs, ws_parameters['batch_size'],
                                  seed=seed, log_results=log_results)
    parse_results(ws_results)
    print_divider()

    print('Evaluating Weight Sharing + Auxiliary Loss Model')
    log_model_information(weight_sharing_aux_loss, wsal_parameters)
    wsal_results = model_evaluation_with_auxiliary_loss(weight_sharing_aux_loss(wsal_parameters),
                                                        train_with_auxiliary_loss, wsal_parameters["aux_loss_weight"],
                                                        rounds, epochs, wsal_parameters['batch_size'], seed=seed,
                                                        log_results=log_results)
    parse_results(wsal_results)
    print_divider()

    model_test_errors = {
        'B1': [round_results[3] for round_results in baseline_1_results],
        'B2': [round_results[3] for round_results in baseline_2_results],
        'WS': [round_results[3] for round_results in ws_results],
        'WSAL': [round_results[3] for round_results in wsal_results]
    }

    # Create box plot
    error_rate_box_plots(model_test_errors)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Finds optimal hyperparameters for models')
    p.add_argument('--rounds', metavar='-r', type=int, default=10,
                   help='The number of times to train the models on different generated datasets.')
    p.add_argument('--log_rounds', action='store_true',
                   help='Flag indicating to log the results for each round of training to the console')
    args = p.parse_args()

    evaluate_models(args.rounds, args.log_rounds)
