This directory contains the codebase for the non-expansive network trained for CIFAR-100.

The collected files are:
- *main.ipynb* that can be run as it is, and contains the lines of code training the network, and plotting the adversarial robustness plots.
- *CertifiedLip_margin_*_cifar10.pt* are files containing the models we trained on cifar-10 and used as initialisation for this case.
- *CertifiedLip_margin_*.pt*, re files containing the models we trained in case it is not desired to train new ones.
- *updateMargin_*.txt* are files containing the obtained robust accuracies for the different values of the *margin* parameter.
- *network.py* is the script implementing the network architecture, having all non-expansive dynamical blocks.
- *training.py* is the script implementing the training routine, including the projection steps to impose the weight constraints
- *multiClassHinge.py* is the script implementing the multi class hinge loss function.
- *utils.py* is the script containing methods useful to define the neural networks in all the directories. 