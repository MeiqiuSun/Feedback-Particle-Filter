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
from FPF import FPF, Model, Galerkin_sympy, Figure

class Oscillator(object):
    def __init__(self, freq_range=[0.9,1.1], amp_range=[[0.9,1.1]], theta_range=[0,2*np.pi], sigma_freq=0.1, sigma_amp=[0.1], sigma_theta=0, tolerance=0):
        assert len(amp_range)==len(sigma_amp), "length of amp_range ({}) and length of sigma_amp ({}) do not match (should all be m)".format(len(amp_range), len(sigma_amp))
        self.m = len(amp_range)
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.theta_range = theta_range
        self.sigma_freq = sigma_freq
        self.sigma_amp = sigma_amp
        self.sigma_theta = sigma_theta
        self.tolerance = tolerance
    
    def get_gain(self):
        pass

class Model_Oscillator(Model):
    def __init__(self, model_type, sigma_W):
        d = 2*len(model_type.amp_range) + 1
        states = ['theta'] * d
        states[1] = 'omega'
        states_label = [r'$\theta\text{ [rad]}$'] * d
        states_label[1] = r'$\omega\text{ [Hz]}$'
        
        X0_range = [model_type.theta_range] * d
        X0_range[1] = model_type.freq_range
        sigma_V = [model_type.sigma_theta] * d
        sigma_V[1] = model_type.sigma_freq
        
        for m in range(len(model_type.amp_range)):
            if m == 0:
                states[2] = 'a{}'.format(1)
                states_label[2] = r'$a_{}$'.format(1)
                X0_range[2] = model_type.amp_range[0]
                sigma_V[2] = model_type.sigma_amp[0]
            else:
                states[2*m+1] = 'a{}'.format(m+1)
                states[2*m+2] = 'b{}'.format(m+1)
                states_label[2*m+1] = r'$a_{}$'.format(m+1)
                states_label[2*m+2] = r'$b_{}$'.format(m+1)
                X0_range[2*m+1] = model_type.amp_range[m]
                X0_range[2*m+2] = model_type.amp_range[m]
                sigma_V[2*m+1] = model_type.sigma_amp[m]
                sigma_V[2*m+2] = model_type.sigma_amp[m]
        self.tolerance = model_type.tolerance
        Model.__init__(self, m=len(model_type.amp_range), d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, sigma_V=sigma_V, sigma_W=sigma_W)
    
    def states_constraints(self, X):
        X[X[:,2]<-self.tolerance,0] += np.pi
        X[X[:,2]<-self.tolerance,2] = -X[X[:,2]<-self.tolerance,2]
        for m in range(self.m):
            X[(X[:,2*m+1]<-self.tolerance) & (X[:,2*m+2]<-self.tolerance),0] += np.pi
            X[(X[:,2*m+1]<-self.tolerance) & (X[:,2*m+2]<-self.tolerance),2*m+1:2*m+3] = \
                -X[(X[:,2*m+1]<-self.tolerance) & (X[:,2*m+2]<-self.tolerance),2*m+1:2*m+3]
            X[X[:,2*m+1]<-self.tolerance,0] = np.pi-X[X[:,2*m+1]<-self.tolerance,0]
            X[X[:,2*m+1]<-self.tolerance,2*m+1] = -X[X[:,2*m+1]<-self.tolerance,2*m+1]
            X[X[:,2*m+2]<-self.tolerance,0] = -X[X[:,2*m+2]<-self.tolerance,0]
            X[X[:,2*m+2]<-self.tolerance,2*m+2] = -X[X[:,2*m+2]<-self.tolerance,2*m+2]
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
        dXdt[:,0] = 2*np.pi*X[:,1]
        return dXdt

    def h(self, X):
        Y = np.zeros((X.shape[0], self.m))
        Y[:,0] = X[:,2] * np.cos(X[:,0])
        Y[:,1:] = np.einsum('im,i->im', X[:,3::2], np.cos(X[:,0])) - np.einsum('im,i->im', X[:,4::2], np.sin(X[:,0]))
        return Y

class Galerkin_Oscillator(Galerkin_sympy):
    """Galerkin_Oscillator: An approximation in finite element method:
            states: [ theta, omega, a1, a2, b2, ..., aM, bM]
            base functions: [ omega, a1*cos(theta), a1*sin(theta), a2*cos(theta), b2*sin(theta), ..., aM*cos(theta), bM*sin(theta)] 
        Initialize = Galerkin_Oscillator(states, M)
            states: 1D list of string with length d, name of each state
            M: int, number of observations
            L: int, number of base functions
        Members = 
            psi_list: 1D list of string with length L, sympy expression of each base function
    """
    def __init__(self, states, M):
        L = 2*M+1
        psi_list = [None for l in range(L)]

        # initialize psi
        psi_list[0] = 'self.omega'
        for m in range(M):
            psi_list[2*m+1] = 'self.a{}*sympy.cos(self.theta)'.format(m+1)
            if m==0:
                psi_list[2*m+2] = 'self.a{}*sympy.sin(self.theta)'.format(m+1)
            else:
                psi_list[2*m+2] = 'self.b{}*sympy.sin(self.theta)'.format(m+1)
        
        Galerkin_sympy.__init__(self, states, M, psi_list)
        return

class FOscillator(object):
    def __init__(self, freq=1., amp_range=[[0.9,1.1]], theta_range=[0,2*np.pi], sigma_theta=0.1, sigma_amp=[0.1], tolerance=0):
        assert len(amp_range)==len(sigma_amp), "length of amp_range ({}) and length of sigma_amp ({}) do not match (should all be m)".format(len(amp_range), len(sigma_amp))
        self.m = len(amp_range)
        self.freq = freq
        self.amp_range = amp_range
        self.theta_range = theta_range
        self.sigma_theta = sigma_theta
        self.sigma_amp = sigma_amp
        self.tolerance = tolerance
    
    def get_gain(self):
        pass

class Model_FOscillator(Model):
    def __init__(self, model_type, sigma_W):
        d = 2*len(model_type.amp_range)
        states = ['theta'] * d
        states_label = [r'$\theta\text{ [rad]}$'] * d
        
        self.freq = model_type.freq
        X0_range = [model_type.theta_range] * d
        sigma_V = [model_type.sigma_theta] * d
        
        for m in range(len(model_type.amp_range)):
            if m == 0:
                states[1] = 'a{}'.format(1)
                states_label[1] = r'$a_{}$'.format(1)
                X0_range[1] = model_type.amp_range[0]
                sigma_V[1] = model_type.sigma_amp[0]
            else:
                states[2*m] = 'a{}'.format(m+1)
                states[2*m+1] = 'b{}'.format(m+1)
                states_label[2*m] = r'$a_{}$'.format(m+1)
                states_label[2*m+1] = r'$b_{}$'.format(m+1)
                X0_range[2*m] = model_type.amp_range[m]
                X0_range[2*m+1] = model_type.amp_range[m]
                sigma_V[2*m] = model_type.sigma_amp[m]
                sigma_V[2*m+1] = model_type.sigma_amp[m]
        self.tolerance = model_type.tolerance

        Model.__init__(self, m=len(model_type.amp_range), d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, sigma_V=sigma_V, sigma_W=sigma_W)
        return
    
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
        dXdt[:,0] = 2*np.pi*self.freq
        return dXdt

    def h(self, X):
        Y = np.zeros((X.shape[0], self.m))
        Y[:,0] = X[:,1] * np.cos(X[:,0])
        Y[:,1:] = np.einsum('im,i->im', X[:,2::2], np.cos(X[:,0])) - np.einsum('im,i->im', X[:,3::2], np.sin(X[:,0]))
        return Y

class Galerkin_FOscillator(Galerkin_sympy):
    """Galerkin_FOscillator: An approximation in finite element method:
            states: [ theta, a1, a2, b2, ..., aM, bM]
            base functions: [ a1*cos(theta), a1*sin(theta), a2*cos(theta), b2*sin(theta), ..., aM*cos(theta), bM*sin(theta)] 
        Initialize = Galerkin_FOscillator(states, M)
            states: 1D list of string with length d, name of each state
            M: int, number of observations
            L: int, number of base functions
        Members = 
            psi_list: 1D list of string with length L, sympy expression of each base function
    """
    def __init__(self, states, M):
        L = 2*M
        psi_list = [None for l in range(L)]

        # initialize psi
        for m in range(M):
            psi_list[2*m] = 'self.a{}*sympy.cos(self.theta)'.format(m+1)
            if m==0:
                psi_list[2*m+1] = 'self.a{}*sympy.sin(self.theta)'.format(m+1)
            else:
                psi_list[2*m+1] = 'self.b{}*sympy.sin(self.theta)'.format(m+1)
        
        Galerkin_sympy.__init__(self, states, M, psi_list)
        return

class ROscillator(object):
    def __init__(self, freq_range=[0.9,1.1], amp=[1], theta_range=[0,2*np.pi], phase_range=[], sigma_freq=0.1, sigma_phase=[]):
        assert len(amp)==(len(phase_range)+1)==(len(sigma_phase)+1), \
            "length of amp ({}) length of phase_range +1 ({}) and length of sigma_phase +1 ({}) do not match (should all be m)".format(\
            len(amp), len(phase_range)+1, len(sigma_phase)+1)
        self.m = len(amp)
        self.freq_range = freq_range
        self.amp = amp
        self.phase_range = phase_range
        self.theta_range = theta_range
        self.sigma_freq = sigma_freq
        self.sigma_phase = sigma_phase
    
    def get_gain(self):
        pass

class Model_ROscillator(Model):
    def __init__(self, model_type, sigma_W):
        d = 2+len(model_type.phase_range)
        states = ['theta'] * d
        states[1] = 'omega'
        states_label = [r'$\theta\text{ [rad]}$'] * d
        states_label[1] = r'$\omega\text{ [Hz]}$'
        
        self.amp = np.array(model_type.amp)
        X0_range = [model_type.theta_range] * d
        X0_range[1] = model_type.freq_range
        sigma_V = [model_type.sigma_freq] * d
        sigma_V[0] = 0.
        
        for m in range(len(model_type.phase_range)):
            states[2+m] = 'phi{}'.format(m+1)
            states_label[2+m] = r'$\phi_%d\text{ [Hz]}$' % (m+1,)
            X0_range[2+m] = model_type.phase_range[m]
            sigma_V[2+m] = model_type.sigma_phase[m]
            
        Model.__init__(self, m=len(model_type.phase_range)+1, d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, sigma_V=sigma_V, sigma_W=sigma_W)

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

class Galerkin_ROscillator(Galerkin_sympy):
    """Galerkin_ROscillator: An approximation in finite element method:
            states: [ theta, omega, phi1, phi2, ..., phi(M-1)]
            base functions: [ omega, cos(theta), sin(theta), cos(phi1)*cos(theta), sin(phi1)*sin(theta), ..., cos(phi(M-1))*cos(theta), sin(phi(M-1))*sin(theta)]
        Initialize = Galerkin_Oscillator(states, M)
            states: 1D list of string with length d, name of each state
            M: int, number of observations
            L: int, number of base functions
        Members = 
            psi_list: 1D list of string with length L, sympy expression of each base function
    """
    def __init__(self, states, M):
        L = 2*M+1
        psi_list = [None for l in range(L)]

        psi_list[0] = 'self.omega'
        for m in range(M):
            if m==0:
                psi_list[2*m+1] = 'sympy.cos(self.theta)'
                psi_list[2*m+2] = 'sympy.sin(self.theta)'
            else:
                psi_list[2*m+1] = 'sympy.cos(self.phi{})*sympy.cos(self.theta)'.format(m)
                psi_list[2*m+2] = 'sympy.sin(self.phi{})*sympy.sin(self.theta)'.format(m)
        
        Galerkin_sympy.__init__(self, states, M, psi_list)
        return

class FROscillator(object):
    def __init__(self, freq=1, amp=[1], theta_range=[0,2*np.pi], phase_range=[], sigma_theta=0.1, sigma_phase=[]):
        assert len(amp)==(len(phase_range)+1)==(len(sigma_phase)+1), \
            "length of amp ({}), length of phase_range +1 ({}) and length of sigma_phase +1 ({}) do not match (should all be m)".format( \
                len(amp), len(phase_range)+1, len(sigma_phase)+1)
        self.m = len(amp)
        self.freq = freq
        self.amp = amp
        self.theta_range = theta_range
        self.phase_range = phase_range
        self.sigma_theta = sigma_theta
        self.sigma_phase = sigma_phase

    def get_gain(self):
        pass

class Model_FROscillator(Model):
    def __init__(self, model_type, sigma_W):
        d = 1+len(model_type.phase_range)
        states = ['theta'] * d
        states_label = [r'$\theta\text{ [rad]}$'] * d
        
        self.amp = np.array(model_type.amp)
        self.freq = model_type.freq
        X0_range = [model_type.theta_range] * d
        sigma_V = [model_type.sigma_theta] * d
        sigma_V[0] = 0.
        
        for m in range(len(model_type.phase_range)):
            states[1+m] = 'phi{}'.format(m+1)
            states_label[1+m] = r'$\phi_%d\text{ [rad]}$$' % (m+1,)
            X0_range[1+m] = model_type.phase_range[m]
            sigma_V[1+m] = model_type.sigma_phase[m]
            
        Model.__init__(self, m=len(model_type.phase_range)+1, d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, sigma_V=sigma_V, sigma_W=sigma_W)

    def states_constraints(self, X):
        for d in range(self.d):
            if np.min(X[:,d])>2*np.pi:
                X[:,d] = np.mod(X[:,d], 2*np.pi)
            if np.max(X[:,d])<0:
                X[:,d] = np.mod(X[:,d], 2*np.pi)
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

class Galerkin_FROscillator(Galerkin_sympy):
    """Galerkin_ROscillator: An approximation in finite element method:
            states: [ theta,  phi1, phi2, ..., phi(M-1)]
            base functions: [ cos(theta), sin(theta), cos(phi1)*cos(theta), sin(phi1)*sin(theta), ..., cos(phi(M-1))*cos(theta), sin(phi(M-1))*sin(theta)]
        Initialize = Galerkin_Oscillator(states, M)
            states: 1D list of string with length d, name of each state
            M: int, number of observations
            L: int, number of base functions
        Members = 
            psi_list: 1D list of string with length L, sympy expression of each base function
    """
    def __init__(self, states, M):
        L = 2*M
        psi_list = [None for l in range(L)]
        
        for m in range(M):
            if m==0:
                psi_list[2*m] = 'sympy.cos(self.theta)'
                psi_list[2*m+1] = 'sympy.sin(self.theta)'
            else:
                psi_list[2*m] = 'sympy.cos(self.phi{})*sympy.cos(self.theta)'.format(m)
                psi_list[2*m+1] = 'sympy.sin(self.phi{})*sympy.sin(self.theta)'.format(m)
        
        Galerkin_sympy.__init__(self, states, M, psi_list)
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
            model = Model_Oscillator(model_type=model_type, sigma_W=sigma_W)
            galerkin = Galerkin_Oscillator(model.states, model.m)
        if isinstance(model_type, FOscillator):
            model = Model_FOscillator(model_type=model_type, sigma_W=sigma_W)
            galerkin = Galerkin_FOscillator(model.states, model.m)
        if isinstance(model_type, ROscillator):
            model = Model_ROscillator(model_type=model_type, sigma_W=sigma_W)
            galerkin = Galerkin_ROscillator(model.states, model.m)
        if isinstance(model_type, FROscillator):
            model = Model_FROscillator(model_type=model_type, sigma_W=sigma_W)
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
    T = 20
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
    model_type = Oscillator(freq_range=[0.9, 1.1], amp_range=[[0.9, 1.1],[0.9,1.1]], sigma_amp=[0.01,0.01])
    # model_type = FOscillator(freq=1., amp_range=[[0.9,1.1],[0.9,1.1]], sigma_amp=[0.01, 0.01])
    # model_type = ROscillator(freq_range=[0.9, 1.1], amp=[1,1], phase_range=[[phase_diff-0.1,phase_diff+0.1]], sigma_phase=[0.1], sigma_freq=0.1)
    # model_type = FROscillator(freq=1., amp=[1,1], phase_range=[[0,0]], sigma_theta=0, sigma_phase=[0.1])
    ofpf = OFPF(number_of_particles=Np, model_type=model_type, sigma_W=[0.1, 0.1], dt=dt)
    filtered_signal = ofpf.run(Y.value, show_time=False)
    # print(3*np.pi/4, np.mean(filtered_signal.X[:,1,-1]))
    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_X_mean=True, plot_c=True, get_states=ofpf.get_states)
    figure = Figure(fig_property=fig_property, Y=Y, filtered_signal=filtered_signal)
    figure.plot_figures(show=True)

