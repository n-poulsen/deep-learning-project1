from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

""" File containing all models implemented for Project 1 """


class BaselineMLP(nn.Module):
    """
    Simple MLP, with a variable amount of hidden layers and units.

    Takes as input batches of 2 x 14 x 14 images.
    """

    def __init__(self, hidden_layer_sizes: List[int], activation: torch.nn.Module):
        super().__init__()
        layers = []
        prev_layer_units = 2 * 14 * 14
        # Add hidden layers
        for i, num_layer_units in enumerate(hidden_layer_sizes):
            layers.append((f'Hidden Layer {i + 1}', nn.Linear(prev_layer_units, num_layer_units)))
            layers.append((f'Activation {i + 1}', activation))
            prev_layer_units = num_layer_units

        # Add final layer
        layers.append((f'Final Layer', nn.Linear(prev_layer_units, 2)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x


class CustomLeNet5(nn.Module):
    """
    Simple CNN.
    """

    def __init__(self, input_channels, output_size):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(input_channels, 12, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d((2, 2)),
            nn.ReLU()
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

    def __init__(self, hidden_layers):
        super().__init__()

        self.fc1 = nn.Linear(20, hidden_layers)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.lenet = CustomLeNet5(input_channels=2, output_size=2)

    def forward(self, x):
        x = self.lenet(x)
        return x


class BaselineCNN2(nn.Module):

    def __init__(self):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=2, output_size=20)
        self.classifier = MLPClassifier(hidden_layers=50)

    def forward(self, x):
        x = self.lenet(x)
        x = self.classifier(x)
        return x


class WeightSharingCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=1, output_size=10)
        self.classifier = MLPClassifier(hidden_layers=50)

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

    def __init__(self):
        super().__init__()

        self.lenet = CustomLeNet5(input_channels=1, output_size=10)
        self.classifier = MLPClassifier(hidden_layers=50)

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
