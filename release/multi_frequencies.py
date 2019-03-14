"""
Created on Thu Jan. 31, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct
from FPF import FPF
from Signal import Signal, Sinusoidals
from single_frequency import Figure

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self):
        self.N = 4
        self.M = 1
        self.X0_range = [[0.5,1.5],[0,2*np.pi],[2.5,3.5],[0,2*np.pi]]
        self.min_sigma_B = 0.01
        self.min_sigma_W = 0.5

    def modified_sigma_B(self, sigma_B):
        sigma_B = sigma_B.clip(min=self.min_sigma_B)
        sigma_B[1::2] = 0
        return sigma_B
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        # X[:,3] = np.mod(X[:,3], 2*np.pi)
        return X

    def f(self, X):
        freq_range=[[1.1,1.3],\
                    [3.7,3.9]]
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = np.linspace(freq_range[0][0]*2*np.pi, freq_range[0][1]*2*np.pi, X.shape[0])
        dXdt[:,3] = np.linspace(freq_range[1][0]*2*np.pi, freq_range[1][1]*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1])+X[:,2]*np.cos(X[:,3])

# 4 states (r1, theta1, r2, theta2) 4 base functions [r1*cos(theta1), r1*sin(theta1), r2*cos(theta2), r2*sin(theta2)]
class Galerkin(object):
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 4       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1]),\
                            np.power(X[:,2],1)*np.cos(X[:,3]), np.power(X[:,2],1)*np.sin(X[:,3])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                [            np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.cos(X[:,3]), -np.power(X[:,2],1)*np.sin(X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.sin(X[:,3]),  np.power(X[:,2],1)*np.cos(X[:,3])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),                    -np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [           -np.sin(X[:,1]), -np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),                    -np.sin(X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),           -np.sin(X[:,3]), -np.power(X[:,2],1)*np.cos(X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),                     np.cos(X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.cos(X[:,3]), -np.power(X[:,2],1)*np.sin(X[:,3])]]]
        return np.array(grad_grad_trial_functions)

 # 4 states (r1, theta1, r2, theta2) 4 base functions [r1*cos(theta1), r1*sin(theta1), r2*cos(theta2), r2*sin(theta2)]

# 4 states (r1, theta1, r2, theta2) 8 base functions [r1*cos(theta1), r1*sin(theta1), r2*cos(theta2), r2*sin(theta2), cos(theta1+theta2), sin(theta1+theta2), cos(theta1-theta2), sin(theta1-theta2)]
class Galerkin_test(object):
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 8       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1]), np.power(X[:,2],1)*np.cos(X[:,3]), np.power(X[:,2],1)*np.sin(X[:,3]),\
                            np.cos(X[:,1]+X[:,3]), np.sin(X[:,1]+X[:,3]), np.cos(X[:,1]-X[:,3]), np.sin(X[:,1]-X[:,3])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                [            np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.cos(X[:,3]), -np.power(X[:,2],1)*np.sin(X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.sin(X[:,3]),  np.power(X[:,2],1)*np.cos(X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]+X[:,3]), np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]+X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),              np.cos(X[:,1]+X[:,3]), np.zeros(X[:,0].shape[0]),              np.cos(X[:,1]+X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]-X[:,3]), np.zeros(X[:,0].shape[0]),              np.sin(X[:,1]-X[:,3])],\
                                [ np.zeros(X[:,0].shape[0]),              np.cos(X[:,1]-X[:,3]), np.zeros(X[:,0].shape[0]),             -np.cos(X[:,1]-X[:,3])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),                    -np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [           -np.sin(X[:,1]), -np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     np.cos(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),                    -np.sin(X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),           -np.sin(X[:,3]), -np.power(X[:,2],1)*np.cos(X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),                     np.cos(X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),            np.cos(X[:,3]), -np.power(X[:,2],1)*np.sin(X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.cos(X[:,1]+X[:,3]), np.zeros(X[:,2].shape[0]),             -np.cos(X[:,1]+X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.cos(X[:,1]+X[:,3]),            np.cos(X[:,3]),             -np.cos(X[:,1]+X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]+X[:,3]), np.zeros(X[:,2].shape[0]),             -np.sin(X[:,1]+X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]+X[:,3]),            np.cos(X[:,3]),             -np.sin(X[:,1]+X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.cos(X[:,1]-X[:,3]), np.zeros(X[:,2].shape[0]),              np.cos(X[:,1]-X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),              np.cos(X[:,1]-X[:,3]),            np.cos(X[:,3]),             -np.cos(X[:,1]-X[:,3])]],\
                                     [[ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),             -np.sin(X[:,1]-X[:,3]), np.zeros(X[:,2].shape[0]),              np.sin(X[:,1]-X[:,3])],\
                                      [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0]),          np.zeros(X[:,3].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),              np.sin(X[:,1]-X[:,3]),            np.cos(X[:,3]),             -np.sin(X[:,1]-X[:,3])]]]
        return np.array(grad_grad_trial_functions)

if __name__ == "__main__":

    T = 5
    fs = 160
    dt = 1/fs
    signal_type = Sinusoidals(dt)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    feedback_particle_filter = FPF(number_of_particles=Np, model=Model(), galerkin=Galerkin(), sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    # feedback_particle_filter = FPF(number_of_particles=Np, model=Model(), galerkin=Galerkin_test(), sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, plot_histogram=True, plot_c=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_matrix(ax, feedback_particle_filter.particles.update_sync_matrix())
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_particle(ax, feedback_particle_filter.particles.check_sync())

    plt.show()