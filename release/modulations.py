"""
Created on Thu Mar. 21, 2019

@author: Heng-Sheng (Hanson) Chang
"""
from __future__ import division
from Signal import Signal

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Modulations(object):
    def __init__(self, dt, rho, X0, amp_r, freq_r, amp_omega, freq_omega, amp_c, freq_c):
        self.SNR = 45
        
        self.dt = dt
        self.rho = rho

        self.X0 = X0
        self.amp_r = np.array(amp_r)
        self.freq_r = np.array(freq_r)
        self.amp_omega = np.array(amp_omega)
        self.freq_omega = np.array(freq_omega)
        self.amp_c = np.array(amp_c)
        self.freq_c = np.array(freq_c)

        self.sigma_B = [0, 0.1, 0, 0]
        self.sigma_W = [self.X0[0]/np.power(10, self.SNR/10)]


    def f(self, X, t):
        X_dot = np.zeros(len(self.sigma_B))
        X_dot[0] = np.sum(self.amp_r*2*np.pi*self.freq_r/self.rho*np.cos(2*np.pi*self.freq_r*t/self.rho))
        X_dot[1] = X[2]
        X_dot[2] = np.sum(2*np.pi*self.amp_omega*2*np.pi*self.freq_omega/self.rho*np.cos(2*np.pi*self.freq_omega*t/self.rho))
        X_dot[3] = np.sum(self.amp_c*2*np.pi*self.freq_c/self.rho*np.cos(2*np.pi*self.freq_c*t/self.rho))
        return X_dot
    
    def h(self, X):
        Y = np.zeros(len(self.sigma_W))
        Y[0] = X[0]*np.sin(X[1])+X[3]
        return Y

if __name__ == "__main__":
    T = 20.
    fs = 160
    dt = 1/fs
    rho = 10
    
    amp0 = 2
    freq0 = 6
    c0 = 1
    X0 = [amp0, 0, 2*np.pi*freq0, c0]

    amp_r = [1,2]
    freq_r = [1,2]
    amp_omega = 5
    freq_omega = 1
    amp_c = 2
    freq_c = 1
    signal_type = Modulations(dt, rho, X0, amp_r, freq_r, amp_omega, freq_omega, amp_c, freq_c)
    signal = Signal(signal_type=signal_type, T=T)
    
    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)


    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for n in range(signal.X.shape[0]):
        Xn = signal.X[n,:]
        if n == 1:
            Xn = np.mod(Xn, 2*np.pi)
        if n == 2:
            Xn = Xn/(2*np.pi)
        ax.plot(signal.t, Xn, label='$X_{}$'.format(n+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)

    plt.show()
