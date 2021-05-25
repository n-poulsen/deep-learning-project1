from models import baseline_1, baseline_2, weight_sharing, weight_sharing_aux_loss
from helpers import error_rate_box_plots, parse_results, print_divider
from evaluation import model_evaluation, model_evaluation_with_auxiliary_loss
from train import train, train_with_auxiliary_loss


def evaluate_models():
    log_results = True
    rounds = 2
    epochs = 10
    seed = 0

    print(f'\n\nEvaluating models through {rounds} rounds of training.')
    print(f'Models trained for {epochs} epochs.')
    print_divider()

    print('Evaluating Baseline 1 Model')
    baseline_1_parameters = {
        'batch_size': 25,
        'lr': 1e-4,
        'weight_decay': 0.0
    }
    model = baseline_1(baseline_1_parameters)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Batch size:         {baseline_1_parameters["batch_size"]}')
    print(f'  Learning rate:      {baseline_1_parameters["lr"]}'),
    print(f'  Weight Decay:       {baseline_1_parameters["weight_decay"]}'),
    print(f'  Model parameters:   {num_params}')
    print()
    baseline_1_results = model_evaluation(
        baseline_1(baseline_1_parameters),
        train,
        rounds, epochs, baseline_1_parameters['batch_size'], seed=seed, print_round_results=log_results)
    parse_results(baseline_1_results)
    print_divider()

    print('Evaluating Baseline 2 Model')
    baseline_2_parameters = {
        'batch_size': 25,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'hidden_layer_units': 50,
    }
    model = baseline_2(baseline_2_parameters)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Batch size:         {baseline_2_parameters["batch_size"]}')
    print(f'  Learning rate:      {baseline_2_parameters["lr"]}'),
    print(f'  Weight Decay:       {baseline_2_parameters["weight_decay"]}'),
    print(f'  Hidden Layer Units: {baseline_2_parameters["hidden_layer_units"]}'),
    print(f'  Model parameters:   {num_params}')
    print()
    baseline_2_results = model_evaluation(
        baseline_2(baseline_2_parameters),
        train,
        rounds, epochs, baseline_2_parameters['batch_size'], seed=seed, print_round_results=log_results)
    parse_results(baseline_2_results)
    print_divider()

    print('Evaluating Weight Sharing Model')
    ws_parameters = {
        'batch_size': 25,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'hidden_layer_units': 50,
    }
    model = weight_sharing(ws_parameters)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Batch size:         {ws_parameters["batch_size"]}')
    print(f'  Learning rate:      {ws_parameters["lr"]}'),
    print(f'  Weight Decay:       {ws_parameters["weight_decay"]}'),
    print(f'  Hidden Layer Units: {ws_parameters["hidden_layer_units"]}'),
    print(f'  Model parameters:   {num_params}')
    print()
    ws_results = model_evaluation(
        weight_sharing(ws_parameters),
        train,
        rounds, epochs, ws_parameters['batch_size'], seed=seed, print_round_results=log_results)
    parse_results(ws_results)
    print_divider()

    print('Evaluating Weight Sharing + Auxiliary Loss Model')
    wsal_parameters = {
        'batch_size': 25,
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'hidden_layer_units': 50,
    }
    aux_loss_weight = 5.0
    model = weight_sharing_aux_loss(wsal_parameters)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Batch size:         {wsal_parameters["batch_size"]}')
    print(f'  Learning rate:      {wsal_parameters["lr"]}'),
    print(f'  Momentum:           {wsal_parameters["momentum"]}'),
    print(f'  Weight Decay:       {wsal_parameters["weight_decay"]}'),
    print(f'  Hidden Layer Units: {wsal_parameters["hidden_layer_units"]}'),
    print(f'  Aux. Loss Weight:   {aux_loss_weight}')
    print(f'  Model parameters:   {num_params}')
    print()
    wsal_results = model_evaluation_with_auxiliary_loss(
        weight_sharing_aux_loss(wsal_parameters),
        train_with_auxiliary_loss,
        aux_loss_weight,
        rounds, epochs, wsal_parameters['batch_size'], seed=seed, print_round_results=log_results)
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
    evaluate_models()
