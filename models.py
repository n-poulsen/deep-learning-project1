import torch
import torch.nn as nn

""" File containing all models implemented for Project 1 """


class CustomLeNet5(nn.Module):
    """ Modified version of LeNet5 for our MNIST dataset """

    def __init__(self, input_channels: int, output_size: int):
        """
        :param input_channels: The number of channels in the data passed as input to the model
        :param output_size: The number of units output by the model
        """
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # Take into account the 2 max pools
        image_out_shape = (14 // 2) // 2
        self.conv_out_size = 32 * (image_out_shape ** 2)

        self.classifier = nn.Sequential(
            nn.Linear(self.conv_out_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_size)
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = x.view((-1, self.conv_out_size))
        x = self.classifier(x)
        return x


class MLPClassifier(nn.Module):
    """ Simple classifier used to process the output of LeNet5, with a single hidden layer. """

    def __init__(self, hidden_layer_units):
        super().__init__()

        self.fc1 = nn.Linear(20, hidden_layer_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_units, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class BaselineCNN(nn.Module):
    """ First baseline classifier: the two images are passed as 2 channels to LeNet5, which outputs two logits """

    def __init__(self):
        super().__init__()
        self.lenet = CustomLeNet5(input_channels=2, output_size=2)

    def forward(self, x):
        x = self.lenet(x)
        return x


class BaselineCNN2(nn.Module):
    """ First baseline classifier: the two images are passed as 2 channels to LeNet5, which outputs a tensor containing
    20 values. These values are then passed to a simple classifier with a single hidden layer. """

    def __init__(self, hidden_layer_units):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=2, output_size=20)
        self.classifier = MLPClassifier(hidden_layer_units=hidden_layer_units)

    def forward(self, x):
        x = self.lenet(x)
        x = self.classifier(x)
        return x


class WeightSharingCNN(nn.Module):
    """ Classifier taking advantage of weight sharing. The input tensor containing the two images is split in two. Each
    image is processed individually in the custom LeNet5. The output logits for both images are concatenated, and
    passed through a simple classifier with a single hidden layer. """

    def __init__(self, hidden_layer_units):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=1, output_size=10)
        self.classifier = MLPClassifier(hidden_layer_units=hidden_layer_units)

    def forward(self, x):
        # Process first images
        x1 = x[:, 0, :, :].reshape(-1, 1, 14, 14)
        x1 = self.lenet(x1)

        # Process second images
        x2 = x[:, 1, :, :].reshape(-1, 1, 14, 14)
        x2 = self.lenet(x2)

        # Concatenate outputs
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        return x


class WeightSharingAuxLossCNN(nn.Module):
    """ Classifier taking advantage of weight sharing and auxiliary losses. The input tensor containing the two images
    is split in two. Each image is processed individually in the custom LeNet5. The output logits for both images are
    concatenated, and passed through a simple classifier with a single hidden layer. The logits for each image are also
    returned, to allow the addition of an auxiliary loss. """

    def __init__(self, hidden_layer_units):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=1, output_size=10)
        self.classifier = MLPClassifier(hidden_layer_units=hidden_layer_units)

    def forward(self, x):
        # Process first images
        x1 = x[:, 0, :, :].reshape(-1, 1, 14, 14)
        x1 = self.lenet(x1)

        # Process second images
        x2 = x[:, 1, :, :].reshape(-1, 1, 14, 14)
        x2 = self.lenet(x2)

        # Concatenate outputs
        x = torch.cat([x1, x2], dim=1)
        x = self.classifier(x)
        return x, x1, x2


def baseline_1(parameters: dict):
    """
    Creates a function that generates an one of our baseline 1 models, with an SGD optimizer and a Cross Entropy loss
    criterion.

    :param parameters: Parameters for the optimizer. Can contain the parameters 'lr' indicating the learning rate,
        'momentum' for the momentum, and 'weight_decay' for the weight decay. Defaults to PyTorch default parameters.
    :return: a function that when called returns the model, main loss criterion, and optimizer
    """
    lr = parameters.get('lr', 0.01)
    weight_decay = parameters.get('weight_decay', 0.0)

    def generate_baseline_1():
        model = BaselineCNN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer

    return generate_baseline_1


def baseline_2(parameters: dict):
    """
    Creates a function that generates an one of our baseline 2 models, with an SGD optimizer and a Cross Entropy loss
    criterion.

    :param parameters: Parameters for the optimizer and model. Can contain the parameters 'lr' (default: 0.01)
        indicating the learning rate, 'momentum' (default: 0.0), 'weight_decay' (default: 0.0), 'hidden_layer_units'
        (default: 50) for the number of units in the hidden layer of the final classification MLP.
    :return: a function that when called returns the model, main loss criterion, and optimizer
    """
    lr = parameters.get('lr', 0.01)
    weight_decay = parameters.get('weight_decay', 0.0)
    hidden_layer_units = parameters.get('hidden_layer_units', 50)

    def generate_baseline_2():
        model = BaselineCNN2(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer

    return generate_baseline_2


def weight_sharing(parameters: dict):
    """
    Creates a function that generates an one of our weight sharing models, with an SGD optimizer, and a Cross Entropy
    loss criterion.

    :param parameters: Parameters for the optimizer and model. Can contain the parameters 'lr' (default: 0.01)
        indicating the learning rate, 'momentum' (default: 0.0), 'weight_decay' (default: 0.0), 'hidden_layer_units'
        (default: 50) for the number of units in the hidden layer of the final classification MLP.
    :return: a function that when called returns the model, main loss criterion, and optimizer
    """
    lr = parameters.get('lr', 0.01)
    weight_decay = parameters.get('weight_decay', 0.0)
    hidden_layer_units = parameters.get('hidden_layer_units', 50)

    def generate_weight_sharing():
        model = WeightSharingCNN(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer

    return generate_weight_sharing


def weight_sharing_aux_loss(parameters):
    """
    Creates a function that generates an one of our weight sharing models with an auxiliary loss, with an SGD optimizer,
    and a Cross Entropy loss criterion.

    :param parameters: Parameters for the optimizer and model. Can contain the parameters 'lr' (default: 0.01)
        indicating the learning rate, 'momentum' (default: 0.0), 'weight_decay' (default: 0.0), 'hidden_layer_units'
        (default: 50) for the number of units in the hidden layer of the final classification MLP.
    :return: a function that when called returns the model, main loss criterion, auxiliary loss criterion, and optimizer
    """
    lr = parameters.get('lr', 0.01)
    weight_decay = parameters.get('weight_decay', 0.0)
    hidden_layer_units = parameters.get('hidden_layer_units', 50)

    def generate_weight_sharing_aux_loss():

        model = WeightSharingAuxLossCNN(hidden_layer_units=hidden_layer_units)
        criterion = torch.nn.CrossEntropyLoss()
        aux_criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return model, criterion, aux_criterion, optimizer

    return generate_weight_sharing_aux_loss
