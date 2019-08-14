"""
Created on Thu Jun. 20, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import sympy

from Tool import Struct
from Signal import Signal, Sinusoids
from System import LTI
from FPF import FPF, Model, Galerkin, Figure

class Oscillator(object):
    def __init__(self, freq_range, amp_range, sigma_freq=0, sigma_amp=[0.1], tolerance=0):
        assert len(amp_range)==len(sigma_amp), "length of amp_range ({}) and length of sigma_amp ({}) do not match (should all be m)".format(len(amp_range), len(sigma_amp))
        self.m = len(amp_range)
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.sigma_freq = sigma_freq
        self.sigma_amp = sigma_amp
        self.tolerance = tolerance
    
    def get_gain(self):
        pass

class Model_Oscillator(Model):
    def __init__(self, freq_range, amp_range, sigma_freq, sigma_amp, sigma_W, tolerance):
        d = 2*len(amp_range)
        states = ['theta'] * d
        states_label = [r'$\theta$'] * d
        
        self.freq_min = freq_range[0]
        self.freq_max = freq_range[1]
        X0_range = [[0,2*np.pi]] * d
        sigma_V = [sigma_freq] * d
        
        for m in range(len(amp_range)):
            if m == 0:
                states[1] = 'a{}'.format(1)
                states_label[1] = r'$a_{}$'.format(1)
                X0_range[1] = amp_range[0]
                sigma_V[1] = sigma_amp[0]
            else:
                states[2*m] = 'a{}'.format(m+1)
                states[2*m+1] = 'b{}'.format(m+1)
                states_label[2*m] = r'$a_{}$'.format(m+1)
                states_label[2*m+1] = r'$b_{}$'.format(m+1)
                X0_range[2*m] = amp_range[m]
                X0_range[2*m+1] = amp_range[m]
                sigma_V[2*m] = sigma_amp[m]
                sigma_V[2*m+1] = sigma_amp[m]
        self.tolerance = tolerance
        Model.__init__(self, m=len(amp_range), d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, 
                        sigma_V=sigma_V, sigma_W=sigma_W)
    
    def states_constraints(self, X):
        X[X[:,1]<-self.tolerance,0] += np.pi
        X[X[:,1]<-self.tolerance,1] = -X[X[:,1]<-self.tolerance,1]
        for m in range(self.m):
            X[(X[:,2*m]<-self.tolerance) & (X[:,2*m+1]<-self.tolerance),0] += np.pi
            X[(X[:,2*m]<-self.tolerance) & (X[:,2*m+1]<-self.tolerance),2*m:2*m+2] = \
                -X[(X[:,2*m]<-self.tolerance) & (X[:,2*m+1]<-self.tolerance),2*m:2*m+2]
            X[X[:,2*m]<-self.tolerance,0] = np.pi-X[X[:,2*m]<-self.tolerance,0]
            X[X[:,2*m]<-self.tolerance,2*m] = -X[X[:,2*m]<-self.tolerance,2*m]
            X[X[:,2*m+1]<-self.tolerance,0] = -X[X[:,2*m+1]<-self.tolerance,0]
            X[X[:,2*m+1]<-self.tolerance,2*m+1] = -X[X[:,2*m+1]<-self.tolerance,2*m+1]
        return X
    
    def get_states(self, X):
        if X.ndim == 1:
            X[0] = np.mod(X[0], 2*np.pi)
        if X.ndim == 2:
            X[0,:] = np.mod(X[0,:], 2*np.pi)
        if X.ndim == 3:
            X[:,0,:] = np.mod(X[:,0,:], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        Y = np.zeros((X.shape[0], self.m))
        Y[:,0] = X[:,1] * np.cos(X[:,0])
        Y[:,1:] = np.einsum('im,i->im', X[:,2::2], np.cos(X[:,0])) + np.einsum('im,i->im', X[:,3::2], np.sin(X[:,0]))
        return Y

class Galerkin_Oscillator(Galerkin):
    """Galerkin_Oscillator: An approximation in finite element method:
            states: [ theta, a1, a2, b2, ..., am, bm]
            base functions: [ a1*cos(theta), a1*sin(theta), a2*cos(theta), b2*sin(theta), ..., am*cos(theta), bm*sin(theta)] 
        Initialize = Galerkin_Oscillator(states, m)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
        Members = 
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string: string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string: string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(self): return sympy executable string
    """
    def __init__(self, states, m):
        Galerkin.__init__(self, states, m, 2*m)
        
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        for m in range(self.m):
            self.psi_list[2*m] = 'self.a{}*sympy.cos(self.theta)'.format(m+1)
            if m==0:
                self.psi_list[2*m+1] = 'self.a{}*sympy.sin(self.theta)'.format(m+1)
            else:
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
        return

    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

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

class ROscillator(object):
    def __init__(self, freq_range, amp, phase_range, sigma_freq=0, sigma_phase=[0.1]):
        assert len(amp)==(len(phase_range)+1)==(len(sigma_phase)+1), \
            "length of amp ({}) length of phase_range +1 ({}) and length of sigma_phase +1 ({}) do not match (should all be m)".format(\
            len(amp), len(phase_range)+1, len(sigma_phase)+1)
        self.m = len(amp)
        self.freq_range = freq_range
        self.amp = amp
        self.phase_range = phase_range
        self.sigma_freq = sigma_freq
        self.sigma_phase = sigma_phase
    
    def get_gain(self):
        pass

class Model_ROscillator(Model):
    def __init__(self, freq_range, amp, phase_range, sigma_freq, sigma_phase, sigma_W):
        d = 2+len(phase_range)
        states = ['theta'] * d
        states[1] = 'omega'
        states_label = [r'$\theta\text{ [rad]}$'] * d
        states_label[1] = r'$\omega\text{ [Hz]}$'
        
        self.amp = np.array(amp)
        X0_range = [[-np.pi/4,np.pi/4]] * d
        X0_range[1] = freq_range
        sigma_V = [sigma_freq] * d
        sigma_V[0] = 0.
        
        for m in range(len(phase_range)):
            states[2+m] = 'phi{}'.format(m+1)
            states_label[2+m] = r'$\phi_{}$'.format(m+1)
            X0_range[2+m] = phase_range[m]
            sigma_V[2+m] = sigma_phase[m]
            
        Model.__init__(self, m=len(phase_range)+1, d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, 
                        sigma_V=sigma_V, sigma_W=sigma_W)

    def states_constraints(self, X):
        if np.min(X[:,0])>2*np.pi:
            X[:,0] = np.mod(X[:,0], 2*np.pi)
        if np.max(X[:,0])<0:
            X[:,0] = np.mod(X[:,0], 2*np.pi)
        return X

    def get_states(self, X):
        if X.ndim == 1:
            X[0] = np.mod(X[0], 2*np.pi)
            if self.m>1:
                X[2:] = np.mod(X[2:], 2*np.pi)
        if X.ndim == 2:
            X[0,:] = np.mod(X[0,:], 2*np.pi)
            if self.m>1:
                X[2:,:] = np.mod(X[2:,:], 2*np.pi)
        if X.ndim == 3:
            X[:,0,:] = np.mod(X[:,0,:], 2*np.pi)
            if self.m>1:
                X[:,2:,:] = np.mod(X[:,2:,:], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = 2*np.pi*X[:,1]
        return dXdt

    def h(self, X):
        Y = np.zeros((X.shape[0], self.m))
        Y[:,0] = self.amp[0]*np.cos(X[:,0])
        if self.m > 1:
            Y[:,1:] = self.amp[1:]*(np.einsum('im,i->im', np.cos(X[:,2:]), np.cos(X[:,0])) - np.einsum('im,i->im', np.sin(X[:,2:]), np.sin(X[:,0])))
        return Y

class Galerkin_ROscillator(Galerkin):
    """Galerkin_ROscillator: An approximation in finite element method:
            states: [ theta, omega, phi1, phi2, ..., phi(m-1)]
            base functions: [ omega, cos(theta), sin(theta), cos(phi1)*cos(theta), sin(phi1)*sin(theta), ..., cos(phi(m-1))*cos(theta), sin(phi(m-1))*sin(theta)]
        Initialize = Galerkin_Oscillator(states, m)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
        Members = 
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string: string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string: string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(self): return sympy executable string
    """
    def __init__(self, states, m):
        Galerkin.__init__(self, states, m, 2*m+1)
        
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        self.psi_list[0] = 'self.omega'
        for m in range(self.m):
            if m==0:
                self.psi_list[2*m+1] = 'sympy.cos(self.theta)'
                self.psi_list[2*m+2] = 'sympy.sin(self.theta)'
            else:
                self.psi_list[2*m+1] = 'sympy.cos(self.phi{})*sympy.cos(self.theta)'.format(m)
                self.psi_list[2*m+2] = 'sympy.sin(self.phi{})*sympy.sin(self.theta)'.format(m)
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

        return

    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

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

class FROscillator(object):
    def __init__(self, freq, amp, phase_range=[], sigma_freq=0, sigma_phase=[]):
        assert len(amp)==(len(phase_range)+1)==(len(sigma_phase)+1), \
            "length of amp ({}), length of phase_range +1 ({}) and length of sigma_phase +1 ({}) do not match (should all be m)".format( \
                len(amp), len(phase_range)+1, len(sigma_phase)+1)
        self.m = len(amp)
        self.freq = freq
        self.amp = amp
        self.phase_range = phase_range
        self.sigma_freq = sigma_freq
        self.sigma_phase = sigma_phase

    def get_gain(self):
        pass

class Model_FROscillator(Model):
    def __init__(self, amp, freq, phase_range, sigma_freq, sigma_phase, sigma_W):
        d = 1+len(phase_range)
        states = ['theta'] * d
        states_label = [r'$\theta$'] * d
        
        self.amp = np.array(amp)
        self.freq = freq
        X0_range = [[0.,2*np.pi]] * d
        sigma_V = [sigma_freq] * d
        sigma_V[0] = 0.
        
        for m in range(len(phase_range)):
            states[1+m] = 'phi{}'.format(m+1)
            states_label[1+m] = r'$\phi_{}$'.format(m+1)
            X0_range[1+m] = phase_range[m]
            sigma_V[1+m] = sigma_phase[m]
            
        Model.__init__(self, m=len(phase_range)+1, d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, 
                        sigma_V=sigma_V, sigma_W=sigma_W)

    def states_constraints(self, X):
        return X

    def get_states(self, X):
        X = np.mod(X, 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = 2*np.pi*self.freq
        return dXdt

    def h(self, X):
        Y = np.zeros((X.shape[0], self.m)) 
        Y[:,0] = np.cos(X[:,0])
        if self.m > 1:
            Y[:,1:] = np.einsum('im,i->im', np.cos(X[:,1:]), np.cos(X[:,0])) - np.einsum('im,i->im', np.sin(X[:,1:]), np.sin(X[:,0]))
        Y[:,:] = np.einsum('m,im->im', self.amp, Y[:,:])
        return Y

class Galerkin_FROscillator(Galerkin):
    """Galerkin_ROscillator: An approximation in finite element method:
            states: [ theta,  phi1, phi2, ..., phi(m-1)]
            base functions: [ cos(theta), sin(theta), cos(phi1)*cos(theta), sin(phi1)*sin(theta), ..., cos(phi(m-1))*cos(theta), sin(phi(m-1))*sin(theta)]
        Initialize = Galerkin_Oscillator(states, m)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
        Members = 
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string: string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string: string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(self): return sympy executable string
    """
    def __init__(self, states, m):
        Galerkin.__init__(self, states, m, 2*m)
        
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        for m in range(self.m):
            if m==0:
                self.psi_list[2*m] = 'sympy.cos(self.theta)'
                self.psi_list[2*m+1] = 'sympy.sin(self.theta)'
            else:
                self.psi_list[2*m] = 'sympy.cos(self.phi{})*sympy.cos(self.theta)'.format(m)
                self.psi_list[2*m+1] = 'sympy.sin(self.phi{})*sympy.sin(self.theta)'.format(m)
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

        return

    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

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
    def __init__(self, number_of_particles, model_type, sigma_W, dt):
        assert model_type.m==len(sigma_W), "length of sigma_W ({}) should be m ({})".format(len(sigma_W), model_type.m)
        if isinstance(model_type, Oscillator):
            model = Model_Oscillator(freq_range=model_type.freq_range, amp_range=model_type.amp_range,\
                                     sigma_freq=model_type.sigma_freq, sigma_amp=model_type.sigma_amp,\
                                     sigma_W=sigma_W, tolerance=model_type.tolerance)
            galerkin = Galerkin_Oscillator(model.states, model.m)
        if isinstance(model_type, ROscillator):
            model = Model_ROscillator(freq_range=model_type.freq_range, amp=model_type.amp, phase_range=model_type.phase_range,\
                                     sigma_freq=model_type.sigma_freq, sigma_phase=model_type.sigma_phase, sigma_W=sigma_W)
            galerkin = Galerkin_ROscillator(model.states, model.m)
        if isinstance(model_type, FROscillator):
            model = Model_FROscillator(freq=model_type.freq, amp=model_type.amp, phase_range=model_type.phase_range, \
                                        sigma_freq=model_type.sigma_freq, sigma_phase=model_type.sigma_phase, sigma_W=sigma_W)
            galerkin = Galerkin_FROscillator(model.states, model.m)
        self.get_gain = model_type.get_gain
        FPF.__init__(self, number_of_particles, model, galerkin, dt)
    
    # def get_gain(self):
    #     gain = np.zeros(self.m, dtype=complex)
    #     for m in range(self.m):
    #         theta_0 = -np.arctan(-np.mean(self.particles.X[:,1]*np.sin(self.particles.X[:,0]))/np.mean(self.particles.X[:,1]*np.cos(self.particles.X[:,0])))
    #         if m==0:
    #             a_1 = np.mean(self.particles.X[:,1]*np.cos(self.particles.X[:,0]-theta_0))
    #             gain[0] = complex(a_1, 0)
    #         else:
    #             a_m = np.mean(self.particles.X[:,2*m]*np.cos(self.particles.X[:,0]-theta_0)) + \
    #                     np.mean(self.particles.X[:,2*m+1]*np.sin(self.particles.X[:,0]-theta_0))
    #             b_m = -np.mean(self.particles.X[:,2*m]*np.sin(self.particles.X[:,0]-theta_0)) + \
    #                     np.mean(self.particles.X[:,2*m+1]*np.cos(self.particles.X[:,0]-theta_0))
    #             gain[m] = complex(a_m, -b_m)
    #         # gain[m] = complex(np.mean(np.square(self.particles.X[:,2*m+1])),-np.mean(np.square(self.particles.X[:,2*m+2])))
    #     return gain

if __name__ == "__main__":
    T = 10
    sampling_rate = 100 # Hz
    dt = 1./sampling_rate

    # sigma_V = [0.1]
    # signal_type = Sinusoids(dt, amp=[[1]], freq=[1], theta0=[[np.pi/2]], sigma_V=sigma_V, SNR=[20])
    # Y, _ = Signal.create(signal_type=signal_type, T=T)
    
    # Np = 1000
    # # model_type = Oscillator(freq_range=[0.9, 1.1], amp_range=[[0.9,1.1]], sigma_amp=[0], sigma_freq=0.1, tolerance=0)
    # model_type = FROscillator(freq=1., amp=[1], phase_range=[], sigma_freq=0, sigma_phase=[])
    # ofpf = OFPF(number_of_particles=Np, model_type=model_type, sigma_W=[0.1], dt=dt)
    # filtered_signal = ofpf.run(Y.value, show_time=False)

    sigma_V = [0.01, 0.02]
    phase_diff = 3*np.pi/4
    signal_type = Sinusoids(dt, amp=[[1,0],[0,1]], freq=[1,1], theta0=[[0,0],[0, phase_diff]], sigma_V=sigma_V, SNR=[20, 20])
    Y, _ = Signal.create(signal_type=signal_type, T=T)

    Np = 1000
    # model_type = Oscillator(freq_range=[0.9, 1.1], amp_range=[[0.9,1.1]], sigma_amp=[0], sigma_freq=0.1, tolerance=0)
    model_type = ROscillator(freq_range=[0.9, 1.1], amp=[1,1], phase_range=[[phase_diff-0.1,phase_diff+0.1]], sigma_phase=[0.1], sigma_freq=0.1)
    # model_type = FROscillator(freq=1., amp=[1,1], phase_range=[[0,0]], sigma_freq=0, sigma_phase=[0.1])
    ofpf = OFPF(number_of_particles=Np, model_type=model_type, sigma_W=[0.1, 0.1], dt=dt)
    filtered_signal = ofpf.run(Y.value, show_time=False)
    # print(3*np.pi/4, np.mean(filtered_signal.X[:,1,-1]))
    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_X_mean=True, plot_c=True, get_states=ofpf.get_states)
    figure = Figure(fig_property=fig_property, Y=Y, filtered_signal=filtered_signal)
    figure.plot_figures(show=True)

