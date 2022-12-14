#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from NormalizedConvolutions import *
from LipschitzConstant import *
from LocalLip import *
import math
        
def train(model, margin, criterion, optimizer, scheduler, trainloader, testloader, device='cpu',epochs=100,reg=True,a=0.1,constrain = True):
    
    print("Initial regularization : ",model.reg(),"\n\n")
    
    for epoch in range(epochs):
        losses = []
        running_loss = 0
        gamma = .1
            
        lip1 = 0
        lip2 = 0
        mar1 = 0
        mar2 = 0
        
        for j in range(3):
            print(f"epoch {epoch}, rescalings_{j} : ",model.seq[j].rescalings.data)        

        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.cpu(),labels.cpu()).to(device)
            
            if reg:
                loss += gamma * model.reg()
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                return -1
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()
            times = 1

            

            if constrain:
                with torch.no_grad():
                    
                    model.FC[-1].weight.data /= max(1,torch.norm(model.FC[-1].weight.data,2)/1)
                    
                    for k in range(3):
                        if k==0:
                            refLip = 1
                        else:
                            refLip = 1
                            
                        shape = (model.seq[k].ML,model.seq[k].kkL,model.seq[k].kkL)
                        lp,model.seq[k].storedEigVectsLift = compute_spectral_norm(model.seq[k].conv1[0], model.seq[k].storedEigVectsLift.unsqueeze(0), shape)
                        model.seq[k].conv1[0].weight.data /= max(1,lp/refLip)
                            
                        model.seq[k].u.data = constrain(model.seq[k].u.data)
                    
                
            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0

        
        for j in range(3):
            print(f"epoch {epoch}, dts_{j} : ",model.seq[j].u.data)

        correct = 0
        reachMargin = 0
        total = 0
        
        model.eval()
        count = 0
        with torch.no_grad():
            norms = []
            nonort = []
            lift = []
            liftOrt = []
            for r in range(3):
              shape = (model.seq[r].ML,model.seq[r].kkL,model.seq[r].kkL)
              n, _ = compute_spectral_norm(model.seq[r].conv1[0], model.seq[r].storedEigVectsLift.unsqueeze(0), shape, k=5)
              lift.append(n.item())
              liftOrt.append(deconv_orth_dist(torch.transpose(model.seq[r].conv1[0].weight,0,1)).item())
              for s in range(model.seq[r].nlayers):
                n, _ = compute_spectral_norm(model.seq[r].convs[s],model.seq[r].storedEigVects[s:s+1],im_size=(model.seq[r].M,model.seq[r].kk,model.seq[r].kk),k=5,device = model.seq[r].convs[s].weight.device)
                nonort.append(deconv_orth_dist(model.seq[r].convs[s].weight).item())
                norms.append(n.item())
            print(f"Norms: {norms}")
            print(f"Orthogonality violation: {nonort}")
            print(f"Reg term: {model.reg().item()}")
            print(f"Norms lifting layers: {lift}")
            print(f"Lifting orth viol: {liftOrt}")
            print(f"Norm last linear: {torch.norm(model.FC[-1].weight.data,2).item()}")
            count = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


            print('Current accuracy on test points: %d %%' % (100 * correct / total))
        model.train()
        scheduler.step()
    print('Training Done')
    return loss