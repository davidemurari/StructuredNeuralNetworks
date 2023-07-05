# Dynamical systems' based neural networks

Repository for the paper "Dynamical systems' based neural networks". The preprint of the paper can be found at : https://arxiv.org/abs/2210.02373 .

The codebase depends on the dependencies collected in the file "requirement.txt". To install the necessary packages run the script
> pip install -r requirements.txt

The codes are organized into 3 main folders:
1. **Adversarial Robustness** : here there are two subfolders
    - CIFAR-10
    - CIFAR-100

The folder contains the Jupyter notebooks with the Adversarial Experiments based on Foolbox library. Each of these two directories collects the experiments for the following training regimes presented in the manuscript

- Unconstrained ResNet
- Naively constrained ResNet
- Non-expansive network
- Prescribed switching regime
- Flexible switching regime.


We add to each of these subdirectories a short readme file to facilitate running the code.

1. **Experiments Appendix** : here we report the codes for the experiments in Appendix A, where we test the architectures obtained starting with splitting methods introduced in the paper. It is organized in two subfolders, as in the paper.
2. **Mass Preserving Networks** : here we report the experiments for mass preserving neural networks.

Citation key:
@article{celledoni2022dynamical,
  title={Dynamical systems' based neural networks},
  author={Celledoni, Elena and Murari, Davide and Owren, Brynjulf and Sch{\"o}nlieb, Carola-Bibiane and Sherry, Ferdia},
  journal={arXiv preprint arXiv:2210.02373},
  year={2022}
}