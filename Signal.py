"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Signal(object):
    """Signal:
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
    def __init__(self, signal_type=None, T=None):
        if signal_type and T:
            self.dt = signal_type.dt
            self.T = self.dt*int(T/self.dt)
            self.t = np.arange(0, self.T+self.dt, self.dt)

            self.f = signal_type.f
            self.h = signal_type.h
            
            self.sigma_B = np.array(signal_type.sigma_B)
            self.sigma_W = np.array(signal_type.sigma_W)

            self.dX = np.zeros([self.sigma_B.shape[0], self.t.shape[0]])
            self.X = np.zeros(self.dX.shape)
            self.dZ = np.zeros([self.sigma_W.shape[0], self.t.shape[0]])
            self.Y = np.zeros(self.dZ.shape)
            for k in range(self.t.shape[0]):
                # dX = f(X, t)*dt + sigma_B*dB
                if k==0:
                    self.dX[:,k] = np.array(signal_type.X0)
                    self.X[:,k] = np.array(signal_type.X0)
                else:
                    self.dX[:,k] = self.f(self.X[:,k-1], self.t[k-1])*self.dt + self.sigma_B*np.random.normal(0, np.sqrt(self.dt), size=self.sigma_B.shape[0])
                    self.X[:,k] = self.X[:,k-1] + self.dX[:,k]
                # dZ = h(X)*dt + sigma_W*dW
                self.dZ[:,k] = self.h(self.X[:,k])*self.dt + self.sigma_W*np.random.normal(0, np.sqrt(self.dt))
                # Y = dZ/dt = h(X) + white noise with std sigma_W
                self.Y[:,k] = self.dZ[:,k] / self.dt
        else:
            self.dt = None
            self.T = None
            self.t = None
            self.sigma_B = None
            self.sigma_W = None
            self.dX = None
            self.X = None
            self.dZ = None
            self.Y = None
    
    def __add__(self, other):
        assert self.dt==other.dt, "time scales (dt) are different"
        assert self.T==other.T, "time length (T) are different"
        signal = Signal()
        
        signal.t = self.t
        signal.dt = self.dt
        signal.T = self.T

        signal.sigma_B = np.zeros(self.sigma_B.shape[0]+other.sigma_B.shape[0])
        signal.sigma_W = np.zeros(self.sigma_W.shape[0]+other.sigma_W.shape[0])
        signal.dX = np.zeros([signal.sigma_B.shape[0], signal.t.shape[0]])
        signal.X = np.zeros(signal.dX.shape)
        signal.dZ = np.zeros([signal.sigma_W.shape[0], signal.t.shape[0]])
        signal.Y = np.zeros(signal.dZ.shape)

        for n in range(self.sigma_B.shape[0]):
            signal.sigma_B[n] = self.sigma_B[n]
            signal.dX[n,:] = self.dX[n,:]
            signal.X[n,:] = self.X[n,:]
        for n in range(other.sigma_B.shape[0]):
            signal.sigma_B[self.sigma_B.shape[0]+n] = other.sigma_B[n]
            signal.dX[self.sigma_B.shape[0]+n,:] = other.dX[n,:]
            signal.X[self.sigma_B.shape[0]+n,:] = other.X[n,:]
        
        for m in range(self.sigma_W.shape[0]):
            signal.sigma_W[m] = self.sigma_W[m]
            signal.dZ[m,:] = self.dZ[m,:]
            signal.Y[m,:] = self.Y[m,:]
        for m in range(other.sigma_W.shape[0]):
            signal.sigma_W[self.sigma_W.shape[0]+m] = other.sigma_W[m]
            signal.dZ[self.sigma_W.shape[0]+m,:] = other.dZ[m,:]
            signal.Y[self.sigma_W.shape[0]+m,:] = other.Y[m,:]
        
        return signal


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
        # SNR: Signal to Noise ratio
        self.SNR = [45]

        # amplitude
        self.amp = [1]
        # frequency
        self.freq = [1.2]
        # initial state condition [A0, theta0]
        self.X0 = [self.amp[0], 0]
        # sampling time
        self.dt = dt
        # N states of state noises
        self.sigma_B = [0, 0.1]
        # M states of observation noises
        self.sigma_W = np.sqrt(np.sum(np.square(self.amp))/np.power(10, np.array(self.SNR)/10)).tolist()
    
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
    def __init__(self, dt, amp=[1,3], freq=[1.2,3.8], theta0=[0,0]):
        # SNR: Signal to Noise ratio
        self.SNR = [45]
        
        # amplitude
        self.amp = amp
        # frequency
        self.freq = freq
        # initial state condition [A0, theta0, A1, theta1]
        self.X0 = [None]*(2*len(self.amp))
        for n in range(len(self.amp)):
            self.X0[2*n] = self.amp[n]
            self.X0[2*n+1] = theta0[n]
        # sampling time
        self.dt = dt

        # N states of state noises
        self.sigma_B = np.zeros(2*len(self.freq))
        for n in range(len(self.freq)):
            self.sigma_B[2*n+1] = 0.1
        # M states of observation noises
        self.sigma_W = np.sqrt(np.sum(np.square(self.amp))/np.power(10, np.array(self.SNR)/10)).tolist()
    
    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        for n in range(len(self.freq)):
            X_dot[2*n+1] = 2*np.pi*self.freq[n]
        return X_dot
    
    # h(X): observation function maps states(X) to observations(Y)
    def h(self, X):
        Y = np.zeros(len(self.sigma_W))
        Y[0] = 0
        for n in range(len(self.freq)):
            Y[0] += X[2*n]*np.sin(X[2*n+1])
        return Y

if __name__ == "__main__":
    
    T = 10.
    fs = 160
    dt = 1/fs
    signal_types = {'Linear':Linear, 'Sinusoidal':Sinusoidal, 'Sinusoidals':Sinusoidals}
    signal_type = signal_types['Sinusoidals'](dt)
    signal = Signal(signal_type=signal_type, T=T)

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()