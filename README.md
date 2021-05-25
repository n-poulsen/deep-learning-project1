# Project 1 - Classification, weight sharing, auxiliary losses

## Project Description

The objective of this project is to test different architectures to compare two digits visible in a two-channel image.
It aims at showing in particular the impact of weight sharing, and of the use of an auxiliary loss to help the training
of the main objective. It should be implemented with PyTorch only code, in particular without using other external
libraries such as scikit-learn or numpy.

### Data

The goal of this project is to implement a deep network such that, given as input a series of 2×14×14 tensor,
corresponding to pairs of 14 × 14 grayscale images, it predicts for each pair if the first digit is lesser or equal to
the second. The training and test set should be 1,000 pairs each, and the size of the images allows to run experiments
rapidly, even in the VM with a single core and no GPU.

You can generate the data sets to use with the function `generate_pair_sets(N)` defined in the file
`dlc_practical_prologue.py`. This function returns six tensors:

| Name          | Tensor Dimension | Type     | Content
|---------------|------------------:|----------:|----------
|`train_input`  | N x 2 x 14 x 14  | `float32`| Images
|`train_target` | N                | `int64`  | Class to predict, in \{0, 1\}
|`train_classes`| N x 2            | `int64`  | Class of the two digits, in \{0, ..., 9\}
|`test_input`   | N x 2 x 14 x 14  | `float32`| Images
|`test_target`  | N                | `int64`  | Class to predict, in \{0, 1\}
|`test_classes` | N x 2            | `int64`  | Class of the two digits, in \{0, ..., 9\}

### Objective

The goal of the project is to compare different architectures, and assess the performance improvement that can be
achieved through weight sharing, or using auxiliary losses. For the latter, the training can in particular take
advantage of the availability of the classes of the two digits in each pair, beside the Boolean value truly of interest.
All the experiments should be done with 1, 000 pairs for training and test. A convnet with ∼ 70, 000 parameters can be
trained with 25 epochs in the VM in less than 2s per epoch and should achieve ∼ 15% error rate.
Performance estimates provided in your report should be estimated through 10+ rounds for each architecture, where both
data and weight initialization are randomized, and you should provide estimates of standard deviations.

## Implementation

### File Structure

All of our models are defined in the `models.py` file. We also create model-generating functions for each one. These
functions return the model, the loss criterion (and auxiliary loss function for the model using an auxiliary loss) and
optimizer used to train them. They all take a single dictionary as a parameter, which contains hyperparameters needed
for the model, loss and optimizer such as learning rate, momentum, weight decay or number of hidden layers in the MLP
classifier.

We use the `dlc_practical_prologue.py` file (a modified version of it as the `argparse` arguments were annoying me) to
generate data points from MNIST. We use a PyTorch Dataset and DataLoader to feed the data to our models while training 
and testing, which are defined in `data_loader.py`.

We have two methods to train our models, defined in `train.py`. The first, `train(*)`, trains models which don't have an
auxiliary loss. The second, `train_with_auxiliary_loss(*)`, trains models which have an auxiliary loss. Testing methods
are defined in the same file.

We define evaluation methods in `evaluation.py`, which iteratively:
* Generate a training and test set using the `generate_pair_sets` from `dlc_practical_prologue.py`.
* Generate the model using a model-generating function
* Train and evaluate the model on the generated datasets

These methods then compute the mean and standard deviation of the performance of the model.

We create hyperparameter tuning methods in `hyperparameter_tuning.py`. These methods take a dictionary containing
hyperparameters (such as batch size, learning rate, momentum, weight decay, ...), which iteratively:
* Take one possible combination of the parameters, and iteratively:
    * Generate ONLY a training set using the `generate_pair_sets` from `dlc_practical_prologue.py`.
    * Do a 80-20 split of the training set, using the 20\% for validation
    * Generate the model using a model-generating function
    * Train the model and evaluate it on the
* Compute the mean loss of the model on the validation set

The methods then combination of parameters that performed best on the validation sets.

The `test.py` file can be called to train and test the models. The `helpers.py` file contains a few helper methods,
such as printing dividers to the console or computing the mean or standard deviation of models. The `plotting.py` file
contains methods used to create plots for our report.

### Training and Evaluating a Model

```
python test.py --rounds 10
```

## Questions for TAs

Should we:
* We based our architecture on modified LeNet-5
    * Trained with auxiliary loss works best
    * Say that we don't need more powerful models as MNIST is an easy dataset: what is tough is training in the right
    way.
* Try batch normalization (even though our neural network isn't very deep)
    * We can
* Try different activation functions
* Try different optimizers
    * Don't need to
* Adam:
    * Momentum computed per coordinate
    * Not great for vision problems
    * No definite answers
    * People still train with momentum
    * Try SGD with weight decay and momentum
* Can we tune our hyperparameters by generating multiple training sets, our should we generate a single training set and
use k-fold cross validation on it?
    * Yes
