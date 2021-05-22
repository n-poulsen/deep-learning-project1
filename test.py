import torch

import models as models
from evaluation import model_evaluation, model_evaluation_with_auxiliary_loss, parse_results
from train import train, train_with_auxiliary_loss


def print_divider():
    print()
    print("-" * 80)
    print("-" * 80)
    print("-" * 80)
    print()


def baseline_1(lr):

    def generate_baseline_1():
        model = models.BaselineCNN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, criterion, optimizer

    return generate_baseline_1


def baseline_2(lr, hidden_layer_units):

    def generate_baseline_2():
        model = models.BaselineCNN2(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, criterion, optimizer

    return generate_baseline_2


def weight_sharing(lr, hidden_layer_units):

    def generate_weight_sharing():
        model = models.WeightSharingCNN(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, criterion, optimizer

    return generate_weight_sharing


def weight_sharing_aux_loss(lr, hidden_layer_units):

    def generate_weight_sharing_aux_loss():

        model = models.WeightSharingAuxLossCNN(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        aux_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return model, criterion, aux_criterion, optimizer

    return generate_weight_sharing_aux_loss


def evaluate_models():
    rounds = 10
    epochs = 25
    batch_size = 25
    seed = 0

    print(f'Evaluating models through {rounds} rounds of training.')
    print(f'Models trained for {epochs} epochs.')
    print_divider()

    print('Evaluating Baseline 1 Model')
    lr = 0.0001
    model = baseline_1(lr)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Learning rate:      {lr}')
    print(f'  Model parameters:   {num_params}')
    print()
    baseline_1_results = model_evaluation(
        baseline_1(lr),
        train,
        rounds, epochs, batch_size, seed=seed, print_round_results=False)
    parse_results(baseline_1_results)
    print_divider()

    print('Evaluating Baseline 2 Model')
    lr = 0.0001
    hidden_layer_units = 50
    model = baseline_2(lr, hidden_layer_units)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Learning rate:      {lr}')
    print(f'  Hidden Layer Units: {hidden_layer_units}')
    print(f'  Model parameters:   {num_params}')
    print()
    baseline_2_results = model_evaluation(
        baseline_2(lr, hidden_layer_units),
        train,
        rounds, epochs, batch_size, seed=seed, print_round_results=False)
    parse_results(baseline_2_results)
    print_divider()

    print('Evaluating Weight Sharing Model')
    lr = 0.0001
    hidden_layer_units = 50
    model = weight_sharing(lr, hidden_layer_units)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Learning rate:      {lr}')
    print(f'  Hidden Layer Units: {hidden_layer_units}')
    print(f'  Model parameters:   {num_params}')
    print()
    ws_results = model_evaluation(
        weight_sharing(lr, hidden_layer_units),
        train,
        rounds, epochs, batch_size, seed=seed, print_round_results=False)
    parse_results(ws_results)
    print_divider()

    print('Evaluating Weight Sharing + Auxiliary Loss Model')
    lr = 0.001
    hidden_layer_units = 50
    aux_loss_weight = 1.0
    model = weight_sharing_aux_loss(lr, hidden_layer_units)()[0]
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Learning rate:      {lr}')
    print(f'  Hidden Layer Units: {hidden_layer_units}')
    print(f'  Aux. Loss Weight:   {aux_loss_weight}')
    print(f'  Model parameters:   {num_params}')
    print()
    wsal_results = model_evaluation_with_auxiliary_loss(
        weight_sharing_aux_loss(lr, hidden_layer_units),
        train_with_auxiliary_loss,
        aux_loss_weight,
        rounds, epochs, batch_size, seed=seed, print_round_results=False)
    parse_results(wsal_results)
    print_divider()


if __name__ == "__main__":
    evaluate_models()
