This directory contains the codebase for the Baseline Unconstrained ResNet trained for CIFAR-10.

The collected files are:
- *main.ipynb* that can be run as it is, and contains the lines of code training the network, and plotting the adversarial robustness plots.
- *cif10_trained_model_margin_*.pt* are files containing the models we trained, in case it is not desired to train new ones.
- *Cifar10_updateMargin_*.txt* are files containing the obtained robust accuracies for the different values of the *margin* parameter.
- *multiClassHinge.py* is the script implementing the multi class hinge loss function.
- *utils.py* is the script containing methods useful to define the neural networks in all the directories. 