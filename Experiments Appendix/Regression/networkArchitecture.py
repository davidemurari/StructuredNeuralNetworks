import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim 
from utils import sumOne, Positive

class CNNBlock(nn.Module):
    def __init__(self, in_chan, nf, n_layers,k,S):
        super(CNNBlock, self).__init__()

        self.S = S
        self.nlayers = n_layers
        self.nf = nf
        self.chans = in_chan
        self.conv1 = nn.Linear(self.chans, self.nf)

        self.Nskew = int(self.nf * (self.nf-1) / 2)

        #Linear layers for gradient layers
        self.convsAP = nn.ModuleList([nn.Linear(self.nf, self.nf) for i in range(self.nlayers)])
        self.alphas = nn.ParameterList([nn.Parameter(torch.randn(self.nf)) for i in range(self.nlayers)])
        #Linear layers for Spherical layers
        self.convsB = nn.ModuleList([nn.Linear(self.nf, self.nf) for i in range(self.nlayers)])
        self.convsC = nn.ModuleList([nn.Linear(self.nf, self.Nskew,bias=False) for i in range(self.nlayers)])
        self.u = torch.nn.Parameter(data=torch.Tensor(self.nlayers))
        self.u.data.copy_(torch.randn(self.nlayers))
        self.nl = nn.ReLU()


    def buildSkew(self,ff):
        dim = self.nf
        res = torch.zeros((len(ff),dim,dim))
        iu1 = torch.triu_indices(dim,dim,1)
        res[:,iu1[0],iu1[1]] = ff
        res = res - torch.transpose(res,1,2)
        return res

    def sphVF(self, y, ff):
        mat = self.buildSkew(ff)
        return torch.einsum('ijk,ik->ij',mat,y)


    def forward(self, x):
        
        x = self.conv1(x)
        dts = Positive(self.u)

        for i in range(self.nlayers):
            
            #Presnov decomposition

            #Expansive like dynamics
            dt = dts[i]/self.S

            A = self.convsAP[i].weight
            b = self.convsAP[i].bias
            D = torch.diag(self.alphas[i])

            for k in range(self.S):
               x = x + dt * F.linear(self.nl(F.linear(x, A) + b),D@A.T)

            if x.shape[1] > 1:
                B = self.convsB[i]
                C = self.convsC[i]
                for k in range(self.S):   
                    ff = C(self.nl(B(x)))
                    x = x + dt * self.sphVF(x, ff)

        return x

class Network(nn.Module):
    def __init__(self, in_chan, nf1, nf2, nf3, n_l1, n_l2, n_l3, dim=1, S=1, a=0.1):
        super(Network, self).__init__()

        self.input = in_chan
        self.S = S
        self.a = a
        self.nf1 = nf1
        self.nf2 = nf2
        self.nf3 = nf3
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.n_l3 = n_l3
        self.dim = dim #dimension in arrival space

        self.seq = nn.Sequential(
            CNNBlock(self.input,self.nf1,self.n_l1,0,self.S),
            CNNBlock(self.nf1,self.nf2,self.n_l2,1,self.S),
            CNNBlock(self.nf2,self.nf3,self.n_l3,2,self.S),
            nn.Flatten(),
            nn.Linear(self.nf3,self.dim)
        )

    def forward(self,x):
      x =  self.seq(x)
      return x