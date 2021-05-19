import torch
import torch.utils.data as data

""" Contains all methods used to load data, as well as methods to feed data to the training algorithm """


class ImageDataset(data.Dataset):
    """ Dataset used to store the pairs images from MNIST """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, classes: torch.Tensor):
        self.inputs = inputs
        self.targets = targets
        self.classes_1 = classes[:, 0]
        self.classes_2 = classes[:, 1]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> (torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor):
        """
        Returns the data pertaining to entry of the dataset.

        :param index: The index of the entry.
        :return: The size (2, 14, 14) tensor containing the pair of images, the size (1,) tensor containing the target
            for the pair of images, and the size (2,) tensors containing the true class of each image.
        """
        return self.inputs[index], self.targets[index], self.classes_1[index], self.classes_2[index]


def train_loader(inputs: torch.FloatTensor, targets: torch.LongTensor,
                 classes: torch.LongTensor, batch_size: int) -> data.DataLoader:
    """
    Creates a training DataLoader for a set of images. The pairs of images can be processed by batch, and the DataLoader
    shuffles the samples.

    :param inputs: A tensor of size (N, 2, 14, 14) containing the N pairs of images of size (14, 14) in the dataset.
    :param targets: A tensor of size (N,) containing the targets for the N pairs of images given as inputs.
    :param classes: A tensor of size (N, 2) containing the classes of each image in the inputs.
    :param batch_size: The batch size of the DataLoader.
    :return: torch.utils.data.DataLoader. The DataLoader for the given input.
    """
    image_dataset = ImageDataset(inputs, targets, classes)
    return data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True)


def test_loader(inputs: torch.FloatTensor, targets: torch.LongTensor,
                classes: torch.LongTensor) -> torch.utils.data.DataLoader:
    """
    Creates a testing DataLoader for a set of images. The pairs of images are processed one by one, and the DataLoader
    does not shuffle the samples.

    :param inputs: A tensor of size (N, 2, 14, 14) containing the N pairs of images of size (14, 14) in the dataset.
    :param targets: A tensor of size (N,) containing the targets for the N pairs of images given as inputs.
    :param classes: A tensor of size (N, 2) containing the classes of each image in the inputs.
    :return: The DataLoader for the given input.
    """
    image_dataset = ImageDataset(inputs, targets, classes)
    return data.DataLoader(image_dataset, batch_size=1, shuffle=False)
