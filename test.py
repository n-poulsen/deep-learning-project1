import argparse

from models import baseline_1, baseline_2, weight_sharing, weight_sharing_aux_loss
from helpers import parse_results, print_divider, log_model_information
from evaluation import model_evaluation, model_evaluation_with_auxiliary_loss
from plotting import error_rate_box_plots, plot_results

# Hyperparameters used to train the baseline_1 model
baseline_1_parameters = {
    'batch_size': 25,
    'lr': 1e-4,
    'weight_decay': 0.1
}

# Hyperparameters used to train the baseline_2 model
baseline_2_parameters = {
    'batch_size': 25,
    'lr': 1e-3,
    'weight_decay': 0.1,
    'hidden_layer_units': 50,
}

# Hyperparameters used to train the weight sharing model
ws_parameters = {
    'batch_size': 25,
    'lr': 1e-4,
    'weight_decay': 0.1,
    'hidden_layer_units': 50,
}

# Hyperparameters used to train the weight sharing with auxiliary loss model
wsal_parameters = {
    'batch_size': 10,
    'lr': 1e-3,
    'weight_decay': 0.0,
    'hidden_layer_units': 100,
    'aux_loss_weight': 10.0,
}


def evaluate_models(rounds: int, log_results: bool, plot: bool):
    epochs = 30
    seed = 0

    print(f'\n\nEvaluating models through {rounds} rounds of training.')
    print(f'Models trained for {epochs} epochs.')
    print_divider()

    print('Evaluating Baseline 1 Model')
    baseline_1_name = 'B1'
    log_model_information(baseline_1, baseline_1_parameters)
    baseline_1_results = model_evaluation(baseline_1(baseline_1_parameters), rounds, epochs,
                                          baseline_1_parameters['batch_size'], seed=seed, log_results=log_results)
    parse_results(baseline_1_results)
    print_divider()

    print('Evaluating Baseline 2 Model')
    baseline_2_name = 'B2'
    log_model_information(baseline_2, baseline_2_parameters)
    baseline_2_results = model_evaluation(baseline_2(baseline_2_parameters), rounds, epochs,
                                          baseline_2_parameters['batch_size'], seed=seed, log_results=log_results)
    parse_results(baseline_2_results)
    print_divider()

    print('Evaluating Weight Sharing Model')
    ws_name = 'WS'
    log_model_information(weight_sharing, ws_parameters)
    ws_results = model_evaluation(weight_sharing(ws_parameters), rounds, epochs, ws_parameters['batch_size'], seed=seed,
                                  log_results=log_results)
    parse_results(ws_results)
    print_divider()

    print('Evaluating Weight Sharing + Auxiliary Loss Model')
    wsal_name = 'WSAL'
    log_model_information(weight_sharing_aux_loss, wsal_parameters)
    wsal_results = model_evaluation_with_auxiliary_loss(
        weight_sharing_aux_loss(wsal_parameters), wsal_parameters["aux_loss_weight"], rounds, epochs,
        wsal_parameters['batch_size'], seed=seed, log_results=log_results)
    parse_results(wsal_results)
    print_divider()

    model_test_errors = {
        baseline_1_name: [round_results[3] for round_results in baseline_1_results],
        baseline_2_name: [round_results[3] for round_results in baseline_2_results],
        ws_name: [round_results[3] for round_results in ws_results],
        wsal_name: [round_results[3] for round_results in wsal_results]
    }

    if plot:
        # Create plots
        plot_results(baseline_1_results, baseline_1_name)
        plot_results(baseline_2_results, baseline_2_name)
        plot_results(ws_results, ws_name)
        plot_results(wsal_results, wsal_name)
        # Create box plot
        error_rate_box_plots(model_test_errors)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Generate and test the models')
    p.add_argument('--rounds', type=int, default=1,
                   help='The number of times to train the models on different generated datasets.')
    p.add_argument('--log_rounds', action='store_true',
                   help='Flag indicating to log the results for each round of training to the console')
    p.add_argument('--plot', action='store_true',
                   help='Flag indicating to display plots')
    args = p.parse_args()

    evaluate_models(args.rounds, args.log_rounds, args.plot)
