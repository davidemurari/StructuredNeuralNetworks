import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from utils import Positive, sumOne, constrain

def train(model, criterion, optimizer, scheduler, trainloader, epochs=100,a=0.1,S=1,reg=True):
    #contracts tells if we want to constrain the time steps
    
    for epoch in range(epochs):
        losses = []
        running_loss = 0
        for i, inp in enumerate(trainloader):
            inputs, labels = inp
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()
            
            '''with torch.no_grad():
                model.seq[-1].weight.data *= model.M/4 / max(1,torch.norm(model.seq[-1].weight.data))
                for k in range(3):
                    model.seq[k].conv1.weight.data *= model.M/4 / max(1,torch.norm(model.seq[k].conv1.weight.data))
                    for j in np.arange(0,model.seq[k].nlayers,2):
                        model.seq[k].convsE[j].weight.data /= max(1,torch.norm(model.seq[k].convs[j].weight.data))'''
            
            if i%10 == 0 and i > 0:
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0
        
        scheduler.step()
    print('Training Done')
    return loss