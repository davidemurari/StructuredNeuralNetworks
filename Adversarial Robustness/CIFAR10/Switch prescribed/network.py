import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class CNNBlock(nn.Module):
    def __init__(self, in_chan, nf, n_layers,maxpool,S,a,k):
        super(CNNBlock, self).__init__()

        self.nlayers = n_layers
        self.nf = nf
        self.chans = in_chan
        self.maxpool = maxpool
        self.a = a
        self.S = S
        self.k = k #block where we are in the global ResNet
        
        self.nl = nn.LeakyReLU(negative_slope=self.a)

        self.u = torch.nn.Parameter(torch.rand(self.nlayers))

        self.conv1 = conv_block(self.chans, self.nf,pool=False, bn=False)
        
        self.convs = nn.ModuleList([nn.Conv2d(self.nf, self.nf,kernel_size=3,stride=1,padding=1,bias=True) for i in range(self.nlayers)])

        self.mp = nn.MaxPool2d(2,2)
        self.ap = nn.AvgPool2d(2,2)
        self.dim = 32 #initial size of one feature map of the image

        self.ML = self.chans
        self.kkL = int(32/2**(self.k-1))*(self.k>0) + 32 * (self.k==0) #lift layer
        self.M = self.nf
        self.kk = int(self.dim/2**self.k) #res layer
        
        self.rescalings = nn.Parameter(torch.rand(1+self.nlayers//2))
        

        self.storedEigVects = torch.rand(self.nlayers,self.M,self.kk,self.kk)
        self.storedEigVectsLift = torch.rand(self.ML,self.kkL,self.kkL)

        for i in range(self.nlayers):
            makeDeltaOrthogonal(self.convs[i].weight.data, nn.init.calculate_gain('leaky_relu',self.a))

    def getReg(self,):
        reg = 0
        for i in range(self.nlayers):
            reg += deconv_orth_dist(self.convs[i].weight)
        return reg
        
    def forward(self, x):
        
        self.rescalings.data[:-1] = torch.clip(self.rescalings.data[:-1],min=0,max=1) #rescalings of the expansive time steps
        
        x = torch.relu(self.rescalings[-1] * self.conv1(x)) #lifting layer
        
        dts = Positive(self.u)
        
        for i in np.arange(0,self.nlayers,2):
        
          dte = dts[i]/self.S
          dtc = dts[i+1]/self.S
          Ae = self.convs[i]
          Ac = self.convs[i+1]

          for k in range(self.S):
                x = x + dte * self.rescalings[i//2] * self.nl(Ae(x))
                x = x - dtc * F.conv_transpose2d(self.nl((Ac(x))), Ac.weight, padding=1, stride=1)

        if self.maxpool :
          return self.mp(x)
        else : 
          return x

class Network(nn.Module):
    def __init__(self, in_chan, nf1, nf2, nf3, n_l1, n_l2, n_l3,S=1,a=0.5,M=1):
        super(Network, self).__init__()

        self.input = in_chan
        self.nf1 = nf1
        self.nf2 = nf2
        self.nf3 = nf3
        self.n_l1 = n_l1
        self.n_l2 = n_l2
        self.n_l3 = n_l3
        self.S = S
        self.a = a
        
        self.seq = nn.Sequential(
            CNNBlock(self.input,self.nf1,self.n_l1,True,self.S,self.a,0),
            CNNBlock(self.nf1,self.nf2,self.n_l2,True,self.S,self.a,1),
            CNNBlock(self.nf2,self.nf3,self.n_l3,True,self.S,self.a,2)
            )
        
        self.last = int((self.seq[2].kk/2) ** 2 * self.seq[2].M)
        
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last,10)
          )
        

    def reg(self,):
      return self.seq[0].getReg() + self.seq[1].getReg() + self.seq[2].getReg()

    def forward(self,x):
      x = self.FC(self.seq(x))
      return x
