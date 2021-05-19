from typing import List, Tuple

import torch
import torch.utils.data as data

""" Contains all of the training methods """


def test(model: torch.nn.Module,
         criterion: torch.nn.Module,
         dataloader: data.DataLoader) -> (float, float):
    """
    Tests a model.

    :param model: The trained model to evaluate.
    :param criterion: The loss criterion to use.
    :param dataloader: The DataLoader containing the test data.
    :return: The average loss and error rate of the model on the test set.
    """
    # Put the model into evaluation mode
    model.eval()

    num_errors = 0
    loss = 0

    with torch.no_grad():
        for i, test_data in enumerate(dataloader):
            inputs, targets, _, _ = test_data
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            predictions = torch.argmax(outputs, dim=1)
            num_errors += (predictions != targets).sum().item()

    error_rate = num_errors / len(dataloader.dataset)
    average_loss = loss / len(dataloader)

    # Put the model back into train mode
    model.train()
    return average_loss, error_rate


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          train_data: data.DataLoader,
          test_data: data.DataLoader,
          num_epochs: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model.

    :param model: The model to train.
    :param optimizer: The optimizer to use for training.
    :param criterion: The loss criterion to use.
    :param train_data: The DataLoader containing the training data.
    :param test_data: The DataLoader containing the test data.
    :param num_epochs: The number of epochs for which to train.
    :return: The training loss, training error, test loss and test error after each epoch.
    """
    train_losses, train_errors = [], []
    test_losses, test_errors = [], []
    for epoch in range(num_epochs):

        # Train for one epoch
        for i, batch in enumerate(train_data):
            inputs, targets, _, _ = batch

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute the error rate on the training set
        train_loss, train_error = test(model, criterion, train_data)
        train_losses.append(train_loss)
        train_errors.append(train_error)

        # Compute the error rate on the test
        test_loss, test_error = test(model, criterion, test_data)
        test_losses.append(test_loss)
        test_errors.append(test_error)

    return train_losses, train_errors, test_losses, test_errors


def test_with_auxiliary_loss(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             aux_criterion: torch.nn.Module,
                             aux_loss_weight: float,
                             dataloader: data.DataLoader) -> (float, float):
    """
    Tests a model.

    :param model: The trained model to evaluate.
    :param criterion: The loss criterion to use.
    :param aux_criterion: The auxiliary loss criterion to use.
    :param aux_loss_weight: The weight to apply to the auxiliary loss
    :param dataloader: The DataLoader containing the test data.
    :return: The average loss and error rate of the model on the test set.
    """
    # Put the model into evaluation mode
    model.eval()

    with torch.no_grad():
        num_errors = 0
        loss = 0
        for i, batch in enumerate(dataloader):
            inputs, targets, aux1, aux2 = batch
            outputs, out_aux1, out_aux2 = model(inputs)

            loss = criterion(outputs, targets)
            loss_aux1 = aux_criterion(out_aux1, aux1)
            loss_aux2 = aux_criterion(out_aux2, aux2)
            total_loss = loss + aux_loss_weight * loss_aux1 + aux_loss_weight * loss_aux2

            loss += total_loss.item()
            predictions = torch.argmax(outputs, dim=1)
            num_errors += (predictions != targets).sum().item()

    error_rate = num_errors / len(dataloader.dataset)
    average_loss = loss / len(dataloader)

    # Put the model back into train mode
    model.train()
    return average_loss, error_rate


def train_with_auxiliary_loss(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        aux_criterion: torch.nn.Module,
        train_data: data.DataLoader,
        test_data: data.DataLoader,
        num_epochs: int,
        aux_loss_weight: float = 1.0) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model.

    :param model: The model to train.
    :param optimizer: The optimizer to use for training.
    :param criterion: The loss criterion to use.
    :param aux_criterion: The auxiliary loss criterion
    :param train_data: The DataLoader containing the training data.
    :param test_data: The DataLoader containing the training data.
    :param num_epochs: The number of epochs for which to train.
    :param aux_loss_weight: The weight to apply to the auxiliary loss
    :return: The training loss, training error, test loss and test error after each epoch.
    """
    train_losses, train_errors = [], []
    test_losses, test_errors = [], []
    for epoch in range(num_epochs):

        for i, batch in enumerate(train_data):
            inputs, targets, aux1, aux2 = batch
            outputs, out_aux1, out_aux2 = model(inputs)

            loss = criterion(outputs, targets)
            loss_aux1 = aux_criterion(out_aux1, aux1)
            loss_aux2 = aux_criterion(out_aux2, aux2)
            total_loss = loss + aux_loss_weight * loss_aux1 + aux_loss_weight * loss_aux2

            model.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Compute the error rate on the training set
        train_loss, train_error = test_with_auxiliary_loss(model, criterion, aux_criterion, aux_loss_weight, train_data)
        train_losses.append(train_loss)
        train_errors.append(train_error)

        # Compute the error rate on the test
        test_loss, test_error = test_with_auxiliary_loss(model, criterion, aux_criterion, aux_loss_weight, test_data)
        test_losses.append(test_loss)
        test_errors.append(test_error)

    return train_losses, train_errors, test_losses, test_errors
