import torch
import numpy as np
import torch.nn as nn

def sumOne(u):
  u = Positive(u)
  u = u/torch.sum(u)
  return Positive(u)

def Positive(X):
    return torch.abs(X)

def constrain(u,norms,S=1,a=0.1):
  u = Positive(u)
  for i in np.arange(0,len(u),2):
    nC = norms[i]
    ss = (1-2*u[i+1]/S*a +u[i+1]**2/S**2) * (1+2*u[i]/S*nC**2+u[i]**2/S**2*nC**4)
    while ss>1:
      u[i]*=0.9
      u[i+1]*=0.9
      ss = (1-2*u[i+1]/S*a +u[i+1]**2/S**2) * (1+2*u[i]/S*nC**2+u[i]**2/S**2*nC**4)
  return u