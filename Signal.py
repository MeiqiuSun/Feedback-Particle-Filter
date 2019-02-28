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
        Initialize = Signal(f, h, sigma_B, sigma_W, X0, dt, T)
            f(X, t): state tansition function maps states(X) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            sigma_B: 1D list with legnth Nd, standard deviation of noise of states(X)
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(Y)
            X0: 1D list with length Nd, initial condition of states(X)
            dt: float, size of time step in sec
            T: float or int, end time in sec
        Members =
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            f(X, t): state tansition function maps states(X) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            sigma_B: numpy array with the shape of (Nd,1), standard deviation of noise of states(X)
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(Y)
            X: numpy array with the shape of (Nd,T/dt+1), states in time series
            dX: numpy array with the shape of (Nd,T/dt+1), increment of states in time series
            Y: numpy array with the shape of (M,T/dt+1), observations in time series
            dZ: numpy array with the shape of (M,T/dt+1), increment of integral of observations in time series
    """

    def __init__(self, f, h, sigma_B, sigma_W, X0, dt, T):
        self.dt = dt
        self.T = self.dt*int(T/self.dt)
        self.t = np.arange(0, self.T+self.dt, self.dt)

        self.f = f
        self.h = h
        
        self.sigma_B = np.array(sigma_B)
        self.sigma_W = np.array(sigma_W)
        np.reshape(np.array(X0), [len(X0),-1])
        self.dX = np.zeros([self.sigma_B.shape[0], self.t.shape[0]])
        self.X = np.zeros(self.dX.shape)
        self.dZ = np.zeros([self.sigma_W.shape[0], self.t.shape[0]])
        self.Y = np.zeros(self.dZ.shape)
        for k in range(self.t.shape[0]):
            # dX = f(X, t)*dt + sigma_B*dB
            if k==0:
                self.dX[:,k] = np.array(X0)
                self.X[:,k] = np.array(X0)
            else:
                self.dX[:,k] = self.f(self.X[:,k-1], self.t[k-1])*self.dt + self.sigma_B*np.random.normal(0, np.sqrt(self.dt))
                self.X[:,k] = self.X[:,k-1] + self.dX[:,k]
            # dZ = h(X)*dt + sigma_W*dW
            self.dZ[:,k] = self.h(self.X[:,k])*self.dt + self.sigma_W*np.random.normal(0, np.sqrt(self.dt))
            # Y = dZ/dt = h(X) + white noise with std sigma_W
            self.Y[:,k] = self.dZ[:,k] / self.dt

class Linear(object):
    def __init__(self, dt):
        # N states of state noises
        self.sigma_B = [0.1, 0.1]
        # M states of observation noises
        self.sigma_W = [0.001, 0.001]
        # initial state condition
        self.X0 = [1, 0]
        # sampling time
        self.dt = dt

    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        A = np.array([[ 0,  1],\
                      [-1, -1]])
        return np.matmul(A,X)

    # h(X): observation function maps states(X) to observations(Y)
    def h(self, X):
        H = np.array([[1,0],\
                      [0,1]])
        return np.matmul(H,X)

class Sinusoidal(object):
    def __init__(self, dt):
        # N states of state noises
        self.sigma_B = [0, 0.1]
        # M states of observation noises
        self.sigma_W = [0.001]
        # amplitude
        self.amp = [1]
        # frequency
        self.freq = [1]
        # initial state condition [A0, theta0]
        self.X0 = [self.amp[0], 0]
        # sampling time
        self.dt = dt
    
    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        X_dot[0] = 0
        X_dot[1] = 2*np.pi*self.freq[0]
        return X_dot
    
    # h(X): observation function maps states(X) to observations(Y)
    def h(self, X):
        Y = np.zeros(len(self.sigma_W))
        Y[0] = X[0]*np.sin(X[1])
        return Y

class Sinusoidals(object):
    def __init__(self, dt):
        # N states of state noises
        self.sigma_B = [0, 0.1, 0, 0.1]
        # M states of observation noises
        self.sigma_W = [0.001]
        # amplitude
        self.amp = [1, 3]
        # frequency
        self.freq = [1, 2.5]
        # initial state condition [A0, theta0, A1, theta1]
        self.X0 = [self.amp[0], 0, self.amp[1], 0]
        # sampling time
        self.dt = dt
    
    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        X_dot[1] = 2*np.pi*self.freq[0]
        X_dot[3] = 2*np.pi*self.freq[1]
        return X_dot
    
    # h(X): observation function maps states(X) to observations(Y)
    def h(self, X):
        Y = np.zeros(len(self.sigma_W))
        Y[0] = X[0]*np.sin(X[1]) + X[2]*np.sin(X[3])
        return Y

if __name__ == "__main__":
    
    T = 10.
    fs = 160
    dt = 1/fs
    signal_types = {'Linear':Linear, 'Sinusoidal':Sinusoidal, 'Sinusoidals':Sinusoidals}
    signal_type = signal_types['Sinusoidals'](dt)
    signal = Signal(f=signal_type.f, h=signal_type.h, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, X0=signal_type.X0, dt=dt, T=T)

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()