"""
Created on Thu Jun. 20, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import sympy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from FPF import FPF
from Signal import Signal, Sinusoidal
from Tool import Struct, isiterable, find_limits

class Model(object):
    def __init__(self, m, frequency_range):
        self.d = 2*m+1
        self.m = m
        self.freq_min = frequency_range[0]
        self.freq_max = frequency_range[1]
        self.states = ['theta'] * self.d
        self.states_label = [r'$\theta$'] * self.d
        for m in range(self.m):
            self.states[2*m+1] = 'a{}'.format(m+1)
            self.states[2*m+2] = 'b{}'.format(m+1)
            self.states_label[2*m+1] = r'$a_{}$'.format(m+1)
            self.states_label[2*m+2] = r'$b_{}$'.format(m+1)
    
    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        Y = np.einsum('im,i->im', X[:,1::2], np.cos(X[:,0])) + np.einsum('im,i->im', X[:,2::2], np.sin(X[:,0]))
        return Y

class Galerkin(object):
    """Galerkin approximation in finite element method:
            states: [ theta, a1, b1, a2, b2, ..., am, bm]
            base functions: [ a1*cos(theta), b1*sin(theta), ..., am*cos(theta), bm*sin(theta)] 
        Notation Note:
            Np: int, number of particles
            X: numpy array with the shape of (Np,d), particle states
        Initialize = Galerkin(states, m)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
        Members = 
            d: int, number of states
            m: int, number of observations
            L: int, number of base functions
            states: 1D list of string with length d, name of each state
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string, string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string, string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(): return sympy executable string
            split(X): split states (X) to members
            psi(X): return a numpy array with the shape of (L,Np), base functions
            grad_psi(X): return a numpy array with the shape of (L,d,Np), gradient of base functions
            grad_grad_psi(X): return a numpy array with the shape of (L,d,d,Np), gradient of gradient of base functions
    """
    def __init__(self, states, m):
        self.d = len(states)
        self.m = m
        self.L = 2*self.m
        
        self.states = ['self.'+state for state in states]
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        for m in range(self.m):
            self.psi_list[2*m] = 'self.a{}*sympy.cos(self.theta)'.format(m+1)
            self.psi_list[2*m+1] = 'self.b{}*sympy.sin(self.theta)'.format(m+1)
        self.psi_string += (', '.join(self.psi_list)).replace('sympy','np')
        self.psi_string += ']'
        
        # initialize symbols
        exec(', '.join(self.states)+' = sympy.symbols("'+', '.join(self.states)+'")')
        
        # calculate grad_psi and grad_grad_psi
        self.grad_psi_string = '['
        self.grad_grad_psi_string = '['
        for l in range(self.L):
            self.grad_psi_string += '['
            self.grad_grad_psi_string += '['
            for d1 in range(self.d):
                exec('self.result = sympy.diff('+self.psi_list[l]+','+self.states[d1]+')')
                self.grad_psi_list[l][d1] = self.sympy_executable_result()
                
                self.grad_grad_psi_string += '['
                for d2 in range(self.d):
                    exec('self.result = sympy.diff('+self.grad_psi_list[l][d1]+','+self.states[d2]+')')
                    self.grad_grad_psi_list[l][d1][d2] = self.sympy_executable_result()
                self.grad_grad_psi_string += ((', ').join(self.grad_grad_psi_list[l][d1])).replace('sympy','np')
                self.grad_grad_psi_string += '], '
            
            self.grad_psi_string += ((', ').join(self.grad_psi_list[l])).replace('sympy','np')
            self.grad_psi_string += '], '
            self.grad_grad_psi_string = self.grad_grad_psi_string[:-2] + '], '

        self.grad_psi_string = self.grad_psi_string[:-2]+']'
        self.grad_grad_psi_string = self.grad_grad_psi_string[:-2]+']'

    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

    def split(self, X):
        for d in range(self.d):
            exec(self.states[d]+' = X[:,{}]'.format(d))

    def psi(self, X):
        self.split(X)
        exec("self.result = "+self.psi_string)
        trial_functions = self.result
        return np.array(trial_functions)
        
    def grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_psi_string)
        grad_trial_functions = np.zeros([self.L, self.d, X.shape[0]])
        for l in range(self.L):
            for d in range(self.d):
                grad_trial_functions[l,d,:] = self.result[l][d]
        return grad_trial_functions

    def grad_grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_grad_psi_string)
        grad_grad_trial_functions = np.zeros([self.L, self.d, self.d, X.shape[0]])
        for l in range(self.L):
            for d1 in range(self.d):
                for d2 in range(self.d):
                    grad_grad_trial_functions[l,d1,d2,:] = self.result[l][d1][d2]
        return grad_grad_trial_functions

class OFPF(FPF):
    def __init__(self, number_of_particles, frequency_range, sigma_B, sigma_W, dt):
        model = Model(len(sigma_W), frequency_range)
        galerkin = Galerkin(model.states, model.m)
        FPF.__init__(self, number_of_particles, model, galerkin, sigma_B, sigma_W, dt)

if __name__ == "__main__":
    # m = 2
    # model = Model(m,[1,2])
    # galerkin = Galerkin(model.states, m)
    ofpf = OFPF(100, [1,2], [0.1], [0.1], 0.1)
    # print(galerkin.__dict__.keys())