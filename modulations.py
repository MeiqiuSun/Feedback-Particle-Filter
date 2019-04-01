"""
Created on Thu Mar. 21, 2019

@author: Heng-Sheng (Hanson) Chang
"""
from __future__ import division

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from Signal import Signal
from single_frequency import Model3, Model4, Galerkin34, Galerkin310, Galerkin4, restrictions, set_yaxis
from FPF import FPF, Figure
from Tool import Struct

class Modulations(object):
    def __init__(self, dt, rho, X0, sigma_B, amp_r, freq_r, amp_omega, freq_omega, amp_c, freq_c):
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

        self.sigma_B = sigma_B
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

def plot_test_signal():
    T = 20.
    fs = 160
    dt = 1/fs
    rho = 10
    
    r0 = 2
    freq0 = 6
    c0 = 1
    X0 = [r0, 0, 2*np.pi*freq0, c0]
    sigma_B = [0.1, 0.1, 2*np.pi*0.1, 0.1]

    amp_r = [1,2]
    freq_r = [1,2]
    amp_omega = 5
    freq_omega = 1
    amp_c = 2
    freq_c = 1
    signal_type = Modulations(dt, rho, X0, sigma_B, amp_r, freq_r, amp_omega, freq_omega, amp_c, freq_c)
    signal = Signal(signal_type=signal_type, T=T)
    plot_signal(signal)

def plot_signal(signal):
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

if __name__ == "__main__":
    T = 20.
    fs = 160
    dt = 1/fs
    rho = 10
    
    r0 = 2
    freq0 = 6
    c0 = 1
    X0 = [r0, 0, 2*np.pi*freq0, c0]
    sigma_B = [0, 0.1, 0, 0]

    amp_r = 0
    freq_r = 1
    amp_omega = 0
    freq_omega = 1
    amp_c = 1
    freq_c = 1
    signal_type = Modulations(dt, rho, X0, sigma_B, amp_r, freq_r, amp_omega, freq_omega, amp_c, freq_c)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    freq_range = [freq0-1, freq0+1]
    amp_range = [r0*0.9,r0*1.1]
    c_range = [0, 3]
    model = Model4(amp_range=amp_range, freq_range=freq_range, c_range=c_range, min_sigma_B=0.01, min_sigma_W=0.5)
    galerkin = Galerkin4()
    signal_type.sigma_B[3] = 2
    # signal_type.sigma_B[0] = 0.1
    fpf = FPF(number_of_particles=Np, model=model, galerkin=galerkin, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = fpf.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, particles_ratio=0.01, restrictions=restrictions(model.states), \
                          plot_histogram=False, n_bins = 100, plot_c=False)
    figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
    figs = set_yaxis(figs, model)
    # plt.figure()
    # plt.plot(signal.t, signal.Y[0]-filtered_signal.h_hat[0])
    figs['X'].axes[3].plot(signal.t, filtered_signal.X_mean[3], color='black')
    figs['X'].axes[3].plot(signal.t, signal.X[3], color='red')
    figs['X'].axes[2].plot(signal.t, filtered_signal.X_mean[2]/2/np.pi, color='black')
    figs['X'].axes[2].plot(signal.t, signal.X[2]/2/np.pi, color='red')
    # plt.figure()
    # for i in range(fpf.particles.Np):
    #     plt.plot(signal.t, filtered_signal.dU[i,3])
    # plot_signal(signal)
    plt.show()
