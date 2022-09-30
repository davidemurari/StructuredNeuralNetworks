from generateData import *
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from networkArchitecture import *
from train import train
from matplotlib import cm


N = 2000
dim = 1 
def f(x):
  if torch.is_tensor(x):
    y = x.detach().numpy().copy()
  else:
    y = x.copy()
  #return  np.linalg.norm(y,axis=1).reshape(-1,1)
  return np.abs(y) + np.sin(y) + y**2

Ntrain = int(0.9 * N)
x,y = generate(N, dim, f)
print(x.shape)
print(y.shape)
x_train = x[:Ntrain]
x_test = x[Ntrain:]
y_train = y[:Ntrain]
y_test = y[Ntrain:]

batch_size = 64

class dataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.from_numpy(x.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.float32))
    self.length = self.x.shape[0]

  def __getitem__(self,idx):
    return self.x[idx],self.y[idx] 
  def __len__(self):
    return self.length 

trainset = dataset(x_train,y_train)
testset = dataset(x_test, y_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

device = 'cpu'
out = 1
net = Network(dim,10,10,10,2,2,2,dim=out, S = 1)
net.to(device)

import torch.optim as optim
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
train(net, criterion, optimizer, scheduler, trainloader,epochs=150)

testPoints = torch.from_numpy(x_test.astype(np.float32)).reshape(-1,dim)
labelsTest = torch.from_numpy(y_test.astype(np.float32))
print(f"\n\n Test error: {criterion(net(testPoints),labelsTest)}\n\n")

if dim==1:
  predFunc = lambda x: net(torch.from_numpy(x.astype(np.float32)).unsqueeze(1)).reshape(-1)
  xx = np.linspace(-3,3,1000)
  plt.plot(xx,f(xx),'k-',label="Exact")
  plt.plot(xx,predFunc(xx).detach().numpy(),'r--',label="Predicted")
  plt.legend(fontsize=12)
  plt.xlabel("x",fontsize=12)
  plt.ylabel(r"$f(x),\,\mathcal{NN}(x)$",fontsize=12)
  plt.title(f"Comparison real and predicted function",fontsize=12)
  plt.show();

if dim==2:
  from mpl_toolkits import mplot3d
  predFunc = lambda x,y: net(torch.from_numpy(np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1).astype(np.float32))).reshape(-1)
  x = np.linspace(-2, 2, 30)
  y = np.linspace(-2, 2, 30)
  X, Y = np.meshgrid(x, y)
  Z = predFunc(X, Y).detach().cpu().numpy().reshape(*X.shape)
  ZZ = f(np.concatenate((X.reshape(-1,1), Y.reshape(-1,1)),axis=1)).reshape(*X.shape)
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X, Y, Z,cmap=cm.binary)
  ax.plot_surface(X, Y, ZZ,cmap=cm.coolwarm)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z');
  ax.set_title("Plot of the real and predicted surfaces")
  plt.show()

  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X, Y, np.abs(ZZ-Z),cmap=cm.coolwarm)
  ax.set_xlabel('x',fontsize=12)
  ax.set_ylabel('y',fontsize=12)
  ax.set_zlabel(r'$|error(x,y)|$',fontsize=12);
  ax.set_title("Difference between real and predicted surfaces",fontsize=12)
  plt.show()