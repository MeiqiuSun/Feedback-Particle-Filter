"""
Created on Thu Jan. 31, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from FPF import FPF, Figure, h
from Signal import Signal
from Tool import Struct

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Galerkin(object):
    # Galerkin approximation in finite element method
    def __init__(self):
        # Ne: int, number of trial (base) functions
        self.L = 3    

    # trial (base) functions
    def psi(self, l, X):
        trial_functions = [X[:,0], np.cos(X[:,1]), np.sin(X[:,1])]
        if l==0:
            return np.array(trial_functions)
        else:
            return trial_functions[l-1]

    # gradient of trial (base) functions
    def grad_psi(self, l, X):
        grad_trial_functions = [[  np.ones(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),           -np.sin(X[:,1])],\
                                [ np.zeros(X[:,0].shape[0]),            np.cos(X[:,1])]]
        if l==0:
            return np.array(grad_trial_functions)
        else:
            return np.array(grad_trial_functions[l-1])
    
    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, l, X):
        grad_grad_trial_functions = [[[np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                      [np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]],\
                                     [[np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                      [np.zeros(X[:,0].shape[0]),           -np.cos(X[:,1])]],\
                                     [[np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                      [np.zeros(X[:,0].shape[0]),           -np.sin(X[:,1])]]]
        if l==0:
            return np.array(grad_grad_trial_functions)
        else:
            return np.array(grad_grad_trial_functions[l-1])

if __name__ == "__main__":
    # N states of frequency inputs
    freq = [1.2, 3.8]
    # M-by-N amplitude matrix
    amp = [[10,5]]
    # N states of state noises
    sigma_B = [0.1,0.1]
    # M states of signal noises
    sigma_W = [0.1]

    T = 10.
    dt = 0.001
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    
    N=100
    feedback_particle_filter = FPF(number_of_particles=N, f_min=0.1, f_max=5, sigma_W=sigma_W, dt=dt, h=h)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_theta=True, plot_freq=True, plot_amp=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_matrix(ax, feedback_particle_filter.particles.update_sync_matrix())
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_particle(ax, feedback_particle_filter.particles.check_sync())

    plt.show()