
INTRODUCTION

The python script performs regressions on the CIFAR10 dataset and tests the models. For the selected models and training settings, it outputs representative results and plots the average training loss and testing accuracy over successive training epochs. It saves these figures in .pdf files in the current directory. 2 models are implemented: a 7-layer CNN and a 7-layer fully-connected model (MLP).

85% testing accuracy can be obtained with the CNN without modifying the parameters.


REQUIREMENTS

To run the script, the following are needed:
Python 3, Torch, Matplotlib.


USAGE

Run the script using the Python 3 interpreter. Select the models and set the learning parameters in the main section by modifying the following vatiables:
- Models: dictionary of models
- num_epochs: number of training epochs
- criterion: loss function
- lrs: list of learning rates
- dfs: list of momentums
- weight_decay: weight decay


ACKNOWLEDGEMENT

Data loader from the CS 260 class at UCLA.

Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009. (CIFAR-10 dataset)
