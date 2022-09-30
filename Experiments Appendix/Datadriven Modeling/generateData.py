import numpy as np
import scipy.integrate

def generate(N,M,T,dim,f,Ntrain,noisy=False):
    #N : how many points to generate
    #M : number of time steps (including the initial condition)
    #T : final time
    #dim : dimension of each point
    #f : vector field
    #Ntrain : percentage of points in the training
    X = np.random.rand(N,dim)*2-1 #random uniform in [-1,1]
    NN = int(Ntrain * N)

    time = np.linspace(0,T,M)
    h = time[1] - time[0]
    print(h)
    traj = np.zeros([N,dim,M])
    for i in range(N):
        traj[i,:,:] = scipy.integrate.solve_ivp(f,[0, T],X[i],method='RK45',t_eval=time,rtol=1e-11,atol=1e-13).y

    Xtrain, ytrain = traj[:NN,:,0], traj[:NN,:,1:]
    Xtest, ytest = traj[NN:,:,0], traj[NN:,:,1:]

    if noisy:
        ytrain += np.random.rand(*ytrain.shape)*0.02-0.01 #noisy training trajectories

    return Xtrain, ytrain, Xtest, ytest, h