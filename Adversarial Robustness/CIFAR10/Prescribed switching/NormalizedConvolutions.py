import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import sumOne, Positive
from LipschitzConstant import *

# From https://github.com/JiJingYu/delta_orthogonal_init_pytorch/blob/master/demo.py
def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.linalg.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    weights[:, :, mid1, mid2] = q[:weights.size(0), :weights.size(1)]
    weights.mul_(gain)

#works with a=0.5, S=2
def constrain(u):
    u = Positive(u)
    layers = len(u)
    h_exp = lambda x: (4-2*torch.sqrt(x**2-2*x+4))/torch.sqrt(x**2-2*x+4)
    for i in range(0,layers,2):
        u[i+1] = max(0.3,min(1,u[i+1]))
        u[i] = h_exp(u[i+1])
    return u
    
    
#Convolutional layers
def deconv_orth_dist(kernel, padding = 2, stride = 1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    return torch.norm( output - target )

def conv_orth_dist(kernel, stride = 1):
    [o_c, i_c, w, h] = kernel.shape
    assert (w == h),"Do not support rectangular kernel"
    assert stride<w,"Please use matrix orthgonality instead"
    new_s = stride*(w-1) + w#np.int(2*(half+np.floor(half/stride))+1)
    temp = torch.eye(new_s*new_s*i_c).reshape((new_s*new_s*i_c, i_c, new_s,new_s)).cuda()
    out = (F.conv2d(temp, kernel, stride=stride)).reshape((new_s*new_s*i_c, -1))
    Vmat = out[np.floor(new_s**2/2).astype(int)::new_s**2, :]
    temp= np.zeros((i_c, i_c*new_s**2))
    for i in range(temp.shape[0]):temp[i,np.floor(new_s**2/2).astype(int)+new_s**2*i]=1
    return torch.norm( Vmat@torch.t(out) - torch.from_numpy(temp).float().cuda() )

#Fully connected layers
def orth_dist(mat, stride=None):
    mat = mat.reshape( (mat.shape[0], -1) )
    if mat.shape[0] < mat.shape[1]:
        mat = mat.permute(1,0)
    return torch.norm( torch.t(mat)@mat - torch.eye(mat.shape[1]).cuda())

def power_method(A, A_t, u_init, k=1):
    u = u_init
    for i in range(k):
        v = A(u)
        v /= torch.sqrt(torch.sum(v**2))
        u = A_t(v)
        sigma = torch.sum(u * u)
        u /= torch.sqrt(torch.sum(u**2))
    return sigma, u[0] #so it returns a 3d tensor

def compute_spectral_norm(conv, u_init=None, im_size=(3, 32, 32), k=1,device='cuda'):
    if u_init is None:
        with torch.no_grad():
            u_init = torch.randn(1, *im_size).to(conv.weight.device)
    u_init = u_init.to(conv.weight.device)
    with torch.no_grad():
        return power_method(lambda u: torch.nn.functional.conv2d(u, conv.weight, padding=tuple(v//2 for v in conv.weight.shape[2:])),
                lambda v: torch.nn.functional.conv_transpose2d(v, conv.weight, padding=tuple(v//2 for v in conv.weight.shape[2:])),
                u_init, k)

def conv_block(in_channels, out_channels, pool=False, bn=False):
    if bn:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels)]
    else:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

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
