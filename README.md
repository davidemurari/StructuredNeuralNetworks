# Dynamical systems' based neural networks

Repository for the paper "Dynamical systems' based neural networks". The preprint of the paper can be found at : https://arxiv.org/abs/2210.02373 .

The codes are organized into 3 main folders:
1. **Adversarial Robustness** : here there are two subfolders
    - Baseline ResNet: with the trained baseline Residual Neural Network
    - Lipschitz Constrained: with the Lipschitz constrained trained networks
   The folder contains then a Jupyter notebook with the Adversarial Experiments based on Foolbox.
2. **Experiments Appendix** : here we report the codes for the experiments in Appendix A, where we test the architectures obtained starting with splitting methods introduced in the paper. It is organized in two subfolders, as in the paper.
3. **Mass Preserving Networks** : here we report the experiments for mass preserving neural networks.

Citation key:
@article{celledoni2022dynamical,
  title={Dynamical systems' based neural networks},
  author={Celledoni, Elena and Murari, Davide and Owren, Brynjulf and Sch{\"o}nlieb, Carola-Bibiane and Sherry, Ferdia},
  journal={arXiv preprint arXiv:2210.02373},
  year={2022}
}