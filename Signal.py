"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Signal(object):
    """
    Signal:
        Notation Notes:
            Nd: dimension of states(X)
            M: number of observations(Y)
        Initialize = Signal(a, b, h, sigma_B, sigma_W, X0, u, dt, T)
            a(X): state tansition function maps states(X) to states(X_dot)
            b(U): input tansition function maps inputs(U) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            sigma_B: 1D list with legnth Nd, standard deviation of noise of states(X)
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(Y)
            X0: 1D list with length Nd, initial condition of states(X)
            u(t): input function maps each time(t) to inputs(U)
            dt: float, size of time step in sec
            T: float or int, end time in sec
        Members =
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            a(X): state tansition function maps states(X) to states(X_dot)
            b(U): input tansition function maps inputs(U) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            U: numpy array with the shape of (,T/dt+1), inputs in time series
            sigma_B: numpy array with the shape of (Nd,1), standard deviation of noise of states(X)
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(Y)
            X: numpy array with the shape of (Nd,T/dt+1), states in time series
            dX: numpy array with the shape of (Nd,T/dt+1), increment of states in time series
            Y: numpy array with the shape of (M,T/dt+1), observations in time series
            dZ: numpy array with the shape of (M,T/dt+1), increment of integral of observations in time series
    """

    def __init__(self, a, b, h, sigma_B, sigma_W, X0, u, dt, T):
        self.dt = dt
        self.T = self.dt*int(T/self.dt)
        self.t = np.arange(0, self.T+self.dt, self.dt)

        self.a = a
        self.b = b
        self.h = h
        self.U = u(self.t)
        
        self.sigma_B = np.array(sigma_B)
        self.sigma_W = np.array(sigma_W)
        
        self.X = np.reshape(np.array(X0), [len(X0),-1])*np.ones([self.sigma_B.shape[0], self.t.shape[0]])
        self.dX = np.zeros(self.X.shape)
        self.dZ = np.zeros([self.sigma_W.shape[0], self.t.shape[0]])
        self.Y = np.zeros(self.dZ.shape)
        for k in range(self.t.shape[0]):
            # dX = a(X)*dt + b(U)*dt + sigma_B*dB
            self.dX[:,k] = self.a(self.X[:,k])*self.dt + self.b(self.U[:,k])*self.dt + np.sqrt(self.dt)*np.random.normal(0, self.sigma_B)
            if not k==(self.t.shape[0]-1):
                self.X[:,k+1] = self.X[:,k] + self.dX[:,k]
            # dZ = h(X)*dt + sigma_W*dW
            self.dZ[:,k] = self.h(self.X[:,k])*self.dt + np.sqrt(self.dt)*np.random.normal(0, self.sigma_W)
            # Y = dZ/dt = h(X) + white noise with std sigma_W
            self.Y[:,k] = self.dZ[:,k] / self.dt

class Linear(object):
    def __init__(self):
        # N states of state noises
        self.sigma_B = [0.1, 0.1]
        # M states of signal noises
        self.sigma_W = [0.001]

        self.X0 = [0, 0]

    def h(self, X):
        H = np.array([[1,0]])
        return np.matmul(H,X)

    def a(self, X):
        A = np.array([[ 0, 1],\
                    [-1, -1]])
        return np.matmul(A,X)

    def b(self, U):
        B = np.array([[0],[1]])
        return np.matmul(B,U)

    def u(self, t):
        U = np.reshape(np.zeros(t.shape[0]), [1,t.shape[0]])
        U[0,0] = 1
        return U

if __name__ == "__main__":
    
    T = 10.
    fs = 160
    dt = 1/fs
    signal_type = Linear()
    signal = Signal(a=signal_type.a, b=signal_type.b, h=signal_type.h, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, \
                    X0=signal_type.X0, u=signal_type.u, dt=dt, T=T)

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.X.shape[0]):
        ax.plot(signal.t, signal.X[m,:], label='$X_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()