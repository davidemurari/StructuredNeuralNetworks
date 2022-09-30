import numpy as np
import matplotlib.pyplot as plt 

def generate(N,dim,f):
    #N : how many points to generate
    #dim : dimension of each point
    #f : function to approximate
    X = np.random.randn(N,dim) #* 6 - 3 #random uniform in [-3,3]
    y = f(X)
    y = y + np.random.randn(*y.shape)*0.01 #add random noise
    return X,y