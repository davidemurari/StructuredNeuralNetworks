import numpy as np
import scipy.integrate

def generate(N,M,T,Ntrain,noisy=False):
    #N : how many points to generate
    #M : number of time steps (including the initial condition)
    #T : final time
    #Ntrain : percentage of points in the training

    np.random.seed(0)

    dim = 3
    '''dim = 6

    k1 = 100/3
    k2 = 1/3
    k3 = 50
    k4 = 1/2
    k5 = 10/3
    k6 = 1/10
    k7 = 7/10

    f = lambda t,y: np.array([(-k7-k1*y[1])*y[0] + k2*y[3] + k6*y[5],
        -k1*y[0]*y[1] + k5 * y[2],
        (-k3*y[0]-k5)*y[2] + k2*y[3] + k4*y[4],
        k1*y[0]*y[1]-k2*y[3],
        k3*y[0]*y[2]-k4*y[4],
        k7*y[0]-k6*y[5]
    ])'''

    R0 = 1
    f = lambda t,y: np.array([-R0*y[0]*y[1],
                            R0*y[0]*y[1] - y[1],
                            y[1]
    ])

    X = np.random.rand(N,dim)
    X = X / np.sum(X,axis=1).reshape(-1,1)
    NN = int(Ntrain * N)

    time = np.linspace(0,T,M)
    h = time[1] - time[0]
    print("Time step: ",h)
    traj = np.zeros([N,dim,M])
    for i in range(N):
        traj[i,:,:] = scipy.integrate.solve_ivp(f,[0, T],X[i],method='RK45',t_eval=time,rtol=1e-11,atol=1e-11).y

    Xtrain, ytrain = traj[:NN,:,0], traj[:NN,:,1:]
    Xtest, ytest = traj[NN:,:,0], traj[NN:,:,1:]

    if noisy:
        ytrain += np.random.rand(*ytrain.shape)*0.01 #noisy training trajectories

    return Xtrain, ytrain, Xtest, ytest, h