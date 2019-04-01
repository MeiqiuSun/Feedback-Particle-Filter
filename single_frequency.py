"""
Created on Mon Mar. 11, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF, Figure
from Signal import Signal, Sinusoidal

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Model1(object):           # 1 state (theta)
    def __init__(self, amp, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$\theta$']
        self.N = len(self.states)
        self.M = 1
        self.r = amp
        self.X0_range = [[0,2*np.pi]]
        self.freq_range = freq_range
        self.min_sigma_B = min_sigma_B
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        sigma_B = np.zeros(self.N)
        return sigma_B
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return sigma_W

    def states_constraints(self, X):
        X[:,0] = np.mod(X[:,0], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_range[0]*2*np.pi, self.freq_range[1]*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return self.r*np.cos(X[:,0])

class Model2omega(object):      # 2 states (theta, omega)
    def __init__(self, amp, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$\theta$',r'$\omega$']
        self.N = len(self.states)
        self.M = 1
        self.r = amp
        self.X0_range = [[0,2*np.pi], [freq*2*np.pi for freq in freq_range]]
        self.min_sigma_B = min_sigma_B
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        new_sigma_B = np.zeros(self.N)
        new_sigma_B[0] = sigma_B[1]
        new_sigma_B = new_sigma_B.clip(min=self.min_sigma_B)
        return new_sigma_B
    
    def modified_sigma_W(self, sigma_W):
        new_sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return new_sigma_W

    def states_constraints(self, X):
        # X[:,0] = np.mod(X[:,0], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = X[:,1]
        return dXdt

    def h(self, X):
        return self.r*np.cos(X[:,0]) 

class Model2(object):           # 2 states (r, theta)
    def __init__(self, amp_range, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$r$',r'$\theta$']
        self.N = len(self.states)
        self.M = 1
        self.X0_range = [amp_range, [0,2*np.pi]]
        self.freq_range = freq_range
        self.min_sigma_B = min_sigma_B
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        sigma_B = sigma_B.clip(min=self.min_sigma_B)
        sigma_B[1::2] = 0
        return sigma_B
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = np.linspace(self.freq_range[0]*2*np.pi, self.freq_range[1]*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1]) 

class Model3(object):           # 3 states (r, theta, omega)
    def __init__(self, amp_range, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$r$',r'$\theta$',r'$\omega$']
        self.N = len(self.states)
        self.M = 1
        self.X0_range = [amp_range, [0,2*np.pi], [freq*2*np.pi for freq in freq_range]]
        self.min_sigma_B = min_sigma_B
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        new_sigma_B = np.zeros(self.N)
        for n in range(np.minimum(sigma_B.shape[0], self.N)):
            new_sigma_B[n] = sigma_B[n]
        new_sigma_B = new_sigma_B.clip(min=self.min_sigma_B)
        return new_sigma_B
    
    def modified_sigma_W(self, sigma_W):
        new_sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return new_sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = X[:,2]
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1]) 

class Model4(object):           # 3 states (r, theta, omega, c)
    def __init__(self, amp_range, freq_range, c_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$r$',r'$\theta$',r'$\omega$',r'$c$']
        self.N = len(self.states)
        self.M = 1
        self.X0_range = [amp_range, [0,2*np.pi], [2*np.pi*freq for freq in freq_range], c_range]
        self.min_sigma_B = min_sigma_B
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        new_sigma_B = np.zeros(self.N)
        for n in range(np.minimum(sigma_B.shape[0], self.N)):
            new_sigma_B[n] = sigma_B[n]
        new_sigma_B = new_sigma_B.clip(min=self.min_sigma_B)
        return new_sigma_B
    
    def modified_sigma_W(self, sigma_W):
        new_sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return new_sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = X[:,2]
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1])+X[:,3]

class Galerkin(object):         # 1 state (theta) 2 base functions [cos(theta), sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 2       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.cos(X[:,0]), np.sin(X[:,0])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ -np.sin(X[:,0])],\
                                [  np.cos(X[:,0])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ -np.cos(X[:,0])]],\
                                     [[ -np.sin(X[:,0])]]]
        return np.array(grad_grad_trial_functions)

class GalerkinOmega(object):    # 2 states (theta, omega) 3 base functions [cos(theta), sin(theta), omega]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 3       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.cos(X[:,0]), np.sin(X[:,0]), X[:,1]]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[           -np.sin(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                [            np.cos(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),  np.ones(X[:,1].shape[0])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[           -np.cos(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]],\
                                     [[           -np.sin(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin0(object):        # 2 states (r, theta) 2 base functions [r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 2       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                [ np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1])],\
                                      [           -np.sin(X[:,1]), -np.power(X[:,0],1)*np.cos(X[:,1])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1])],\
                                      [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin1(object):        # 2 states (r, theta) 3 base functions [r, r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 3       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ X[:,0], np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ np.ones(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0])],\
                                [           np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                [           np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1])],\
                                      [           -np.sin(X[:,1]),  -np.power(X[:,0],1)*np.cos(X[:,1])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1])],\
                                      [            np.cos(X[:,1]),  -np.power(X[:,0],1)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin2(object):        # 2 states (r, theta) 3 base functions [r^2/2, r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 3       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],2)/2, np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[         X[:,0],          np.zeros(X[:,1].shape[0])],\
                                [ np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                [ np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[  np.ones(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1])],\
                                      [           -np.sin(X[:,1]),  -np.power(X[:,0],1)*np.cos(X[:,1])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1])],\
                                      [            np.cos(X[:,1]),  -np.power(X[:,0],1)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin3(object):        # 2 states (r, theta) 3 base functions [r^3/6, r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 3       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],3)/6, np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ np.power(X[:,0],2)/2,          np.zeros(X[:,1].shape[0])],\
                                [       np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                [       np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[                    X[:,0],           np.zeros(X[:,1].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1])],\
                                      [           -np.sin(X[:,1]),  -np.power(X[:,0],1)*np.cos(X[:,1])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1])],\
                                      [            np.cos(X[:,1]),  -np.power(X[:,0],1)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin34(object):       # 3 states (r, theta, omega) 4 base functions [r, r*cos(theta), r*sin(theta), omega^p/p!]
    # Galerkin approximation in finite element method
    def __init__(self, power=1):
        # L: int, number of trial (base) functions
        self.L = 4       # trial (base) functions
        self.power = power
        self.factorial = 1
        for p in range(self.power):
            self.factorial = self.factorial*(p+1)
        
        
    def psi(self, X):
        trial_functions = [ X[:,0], np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1]), np.power(X[:,2],self.power)/self.factorial]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[  np.ones(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),                               np.zeros(X[:,2].shape[0])],\
                                [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]),                               np.zeros(X[:,2].shape[0])],\
                                [            np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1]),                               np.zeros(X[:,2].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.power(X[:,2],self.power-1)*self.power/self.factorial]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [           -np.sin(X[:,1]),  -np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [            np.cos(X[:,1]),  -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]),  np.power(X[:,2],self.power-2)*(self.power-1)*self.power/self.factorial if not self.power==1 else np.zeros(X[:,2].shape[0])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin310(object):       # 3 states (r, theta, omega) 10 base functions [r, r^2/2, r^3/6, r^4/24, r*cos(theta), r*sin(theta), omega, omega^2/2, omega^3/6, omega^4/24]
    # Galerkin approximation in finite element method
    def __init__(self, power=1):
        # L: int, number of trial (base) functions
        self.L = 10        
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],1), np.power(X[:,0],2)/2, np.power(X[:,0],3)/6, np.power(X[:,0],4)/24,\
                            np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1]),\
                            np.power(X[:,2],1), np.power(X[:,2],2)/2, np.power(X[:,2],3)/6, np.power(X[:,2],4)/24]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[  np.ones(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                [        np.power(X[:,0],1),          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                [      np.power(X[:,0],2)/2,          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                [      np.power(X[:,0],3)/6,          np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                [            np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),  np.ones(X[:,2].shape[0])],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),        np.power(X[:,2],1)],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),      np.power(X[:,2],2)/2],\
                                [ np.zeros(X[:,0].shape[0]),          np.zeros(X[:,1].shape[0]),      np.power(X[:,2],3)/6]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[  np.ones(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[        np.power(X[:,0],1),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[      np.power(X[:,0],2)/2,           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [           -np.sin(X[:,1]),  -np.power(X[:,0],1)*np.cos(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [            np.cos(X[:,1]),  -np.power(X[:,0],1)*np.sin(X[:,1]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]),  np.ones(X[:,2].shape[0])]],\
                                     [[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]),        np.power(X[:,2],1)]],\
                                     [[ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]), np.zeros(X[:,2].shape[0])],\
                                      [ np.zeros(X[:,0].shape[0]),           np.zeros(X[:,1].shape[0]),      np.power(X[:,2],2)/2]]]
        return np.array(grad_grad_trial_functions)

class Galerkin4(object):       # 3 states (r, theta, omega, c) 4 base functions [r, r*cos(theta), r*sin(theta), omega, c]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 5       # trial (base) functions
    
    def split(self, X):
        r = X[:,0]
        theta = X[:,1]
        omega = X[:,2]
        c = X[:,3]
        zeros = np.zeros(X.shape[0]) 
        ones = np.ones(X.shape[0])
        return r, theta, omega, c, zeros, ones
        
    def psi(self, X):
        r, theta, omega, c, _, _ = self.split(X)
        trial_functions = [ r, r*np.cos(theta), r*np.sin(theta), omega, c]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        r, theta, omega, c, zeros, ones = self.split(X)
        grad_trial_functions = [[          ones,            zeros, zeros, zeros],\
                                [ np.cos(theta), -r*np.sin(theta), zeros, zeros],\
                                [ np.sin(theta),  r*np.cos(theta), zeros, zeros],\
                                [         zeros,            zeros,  ones, zeros],\
                                [         zeros,            zeros, zeros,  ones]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        r, theta, omega, c, zeros, ones = self.split(X)
        grad_grad_trial_functions = [[[          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros]],\
                                     [[          zeros,    -np.sin(theta), zeros, zeros],\
                                      [ -np.sin(theta),  -r*np.cos(theta), zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros]],\
                                     [[          zeros,     np.cos(theta), zeros, zeros],\
                                      [  np.cos(theta),  -r*np.sin(theta), zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros]],\
                                     [[          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros]],\
                                     [[          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros],\
                                      [          zeros,             zeros, zeros, zeros]]]
        return np.array(grad_grad_trial_functions)


def restrictions(states):
    mod = []
    scaling = []
    center = []
    zero = []
    for state in states:
        mod.append(2*np.pi if 'theta' in state else np.nan)
        scaling.append(1/(2*np.pi) if 'omega' in state else 1)
        center.append('r' in state)
        zero.append('omega' in state)
    return Struct(mod=mod, scaling=scaling, center=center, zero=zero)

def set_yaxis(figs, model):
    for n, state in enumerate(model.states):
        if 'X' in figs.keys():
            figs['X'].axes[n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
        if 'histogram' in figs.keys():
            figs['histogram'].axes[n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
        if 'theta' in state:
            if 'X' in figs.keys():
                figs['X'].axes[n].set_yticks(np.linspace(0,2*np.pi,5))
                figs['X'].axes[n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
            if 'histogram' in figs.keys():
                figs['histogram'].axes[n].set_yticks(np.linspace(0,2*np.pi,5))
                figs['histogram'].axes[n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
        if 'omega' in state:
            pass
    return figs

if __name__ == "__main__":

    T = 5
    sampling_rate = 160 # Hz
    dt = 1./sampling_rate
    signal_type = Sinusoidal(dt)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    # freq_range = [1.1,1.3]
    freq_range = [0.1,1.5]
    amp = signal_type.amp[0]
    # model = Model1(amp=amp, freq_range=freq_range)
    # model = Model2omega(amp=amp, freq_range=freq_range)
    amp_range = [0.5,1.5]
    # model = Model2(amp_range=amp_range, freq_range=freq_range)
    model = Model3(amp_range=amp_range, freq_range=freq_range)
    # galerkin = Galerkin()
    galerkin = Galerkin310()
    # galerkin = GalerkinOmega()
    feedback_particle_filter = FPF(number_of_particles=Np, model=model, galerkin=galerkin, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, particles_ratio=0.01, restrictions=restrictions(model.states), \
                          plot_histogram=True, n_bins = 100, plot_c=True)
    figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
    figs = set_yaxis(figs, model)
    plt.show()