import torch
import numpy as np
import torch.nn as nn

def sumOne(u):
  u = Positive(u)
  u = u/torch.sum(u)
  return Positive(u)

def Positive(X):
    return torch.abs(X)