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
