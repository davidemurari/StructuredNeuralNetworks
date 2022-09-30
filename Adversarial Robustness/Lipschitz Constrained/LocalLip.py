import torch
import torch.nn as nn

#We want to approximate empirically the local Lipschitz constant of a function

def LocLip(f,eps,X,initial,device='cpu'):
    #f is the function for which we want the local experimental Lipschitz constant
    #eps is the radius saying how far from the points in X we can go to do the approximation
    #X is the set of points about which we want to compute the constant
    #device is the device where the involved objects are stored
    
    Y = X.requires_grad_().to(device)

    times = 1000
    Appr = initial
    Appr = Appr.to(device)
    for i in range(times):
        o = torch.ones(len(Y)).to(device)
        Appr.requires_grad_()
        quotient = torch.div(torch.linalg.norm(f(Y)-f(Appr),2,dim=(1)),torch.norm(Y-Appr, 2,dim=(1,2,3)))
        func = torch.mean(quotient,axis=0)
        grad = torch.autograd.grad(func,Appr)[0]
        Appr = Appr + eps / 50 * grad
        Appr = Y + torch.einsum('ijkl,i->ijkl',Appr-Y,eps/torch.maximum(o,torch.norm(Appr-Y,2,dim=(1,2,3))))
    
    return Appr,func


def LocMargin(f,eps,X,initial,device='cpu'):
    #f is the function for which we want the local experimental Lipschitz constant
    #eps is the radius saying how far from the points in X we can go to do the approximation
    #X is the set of points about which we want to compute the constant
    #device is the device where the involved objects are stored
    
    Y = X.requires_grad_().to(device)

    times = 10
    Appr = initial
    Appr = Appr.to(device)
    for i in range(times):
        o = torch.ones(len(Y)).to(device)
        Appr.requires_grad_()
        func = torch.mean(-torch.diff(torch.topk(f(Appr),2)[0],dim=1))
        grad = torch.autograd.grad(func,Appr)[0]
        Appr = Appr - eps / 5 * grad #I want the smallest margin possible
        Appr = Y + torch.einsum('ijkl,i->ijkl',Appr-Y,eps/torch.maximum(o,torch.norm(Appr-Y,2,dim=(1,2,3)))) #project on the eps-ball around X
    
    return Appr,func