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
            N: number of states(X)
            M: number of observations(Y)
        Initialize = Signal(freq, amp, sigma_B, sigma_W, dt, T)
            freq: 1D list with length N, list of freqencies in Hz
            amp: 2D list with shape of M-by-N, amplitude parameters for h function in Y=h(X)
            sigma_B: 1D list with legnth N, standard deviation of noise of states(X)
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(Y)
            dt: float, size of time step in sec
            T: float or int, end time in sec
        Members =
            omega: numpy array with shape of (N,), list of freqencies in rad/s
            amp: numpy array with the shape of (M,N), amplitude parameters for h function in Y=h(X)
            sigma_B: numpy array with the shape of (N,1), standard deviation of noise of states(X)
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(Y)
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            X: numpy array with the shape of (N,T/dt+1), states in time series
            dX: numpy array with the shape of (N,T/dt+1), states increment in time series
            Y: numpy array with the shape of (M,T/dt+1), observations in time series
        Methods =
            h(amp, x): observation function maps states(X) to observations(Y)
    """
    
    def __init__(self, freq, amp, sigma_B, sigma_W, dt, T):
        self.omega = 2*np.pi*(np.array(freq))
        self.amp = np.array(amp)
        self.sigma_B = np.reshape(np.array(sigma_B), [self.amp.shape[1],-1])
        self.sigma_W = np.reshape(np.array(sigma_W), [self.amp.shape[0],-1])
        self.dt = dt
        self.T = self.dt*int(T/self.dt)
        self.t = np.arange(0,self.T+self.dt,self.dt)
        self.X = np.zeros([self.sigma_B.shape[0], self.t.shape[0]])
        self.dX = np.zeros(self.X.shape)
        for n in range(self.X.shape[0]):
            """dX = omega*dt + sigma_B*dB"""
            self.dX[n,:] = self.omega[n]*self.dt + self.sigma_B[n]*self.dt*np.random.normal(0,1,self.t.shape[0])
            self.X[n,:] = np.cumsum(self.dX[n,:])
        self.Y = self.h(self.amp, self.X) + self.sigma_W*np.random.normal(0,1,[self.sigma_W.shape[0],self.t.shape[0]])

    def h(self, amp, x):
        """observation function maps states(x) to observations(y)"""
        return np.matmul(amp, np.sin(x))
