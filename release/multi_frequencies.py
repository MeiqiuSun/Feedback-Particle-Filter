"""
Created on Thu Jan. 31, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import sys
from Tool import Struct
# import CFPF
from Signal import Signal, Sinusoidals

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def fpf():
    from FPF import FPF, Figure
    from single_frequency import restrictions, set_yaxis

    class Model2(object):   # 2 states (theta1, theta2)
        def __init__(self, amp, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
            self.states = [r'$\theta_1$',r'$\theta_2$']
            self.N = len(self.states)
            self.M = 1
            self.amp = amp
            self.X0_range = [[0,2*np.pi],[0,2*np.pi]]
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
            # X[:,0] = np.mod(X[:,0], 2*np.pi)
            # X[:,1] = np.mod(X[:,1], 2*np.pi)
            return X

        def f(self, X):
            dXdt = np.zeros(X.shape)
            dXdt[:,0] = np.linspace(self.freq_range[0][0]*2*np.pi, self.freq_range[0][1]*2*np.pi, X.shape[0])
            dXdt[:,1] = np.linspace(self.freq_range[1][0]*2*np.pi, self.freq_range[1][1]*2*np.pi, X.shape[0])
            return dXdt

        def h(self, X):
            return self.amp[0]*np.cos(X[:,0])+self.amp[1]*np.cos(X[:,1])

    class Model4(object):   # 4 states (r1, theta1, r2, theta2)
        def __init__(self, amp_range, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
            self.states = [r'$r_1$',r'$\theta_1$',r'$r_2$',r'$\theta_2$']
            self.N = len(self.states)
            self.M = 1
            self.X0_range = [amp_range[0],[0,2*np.pi],amp_range[1],[0,2*np.pi]]
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
            # X[:,3] = np.mod(X[:,3], 2*np.pi)
            return X

        def f(self, X):
            dXdt = np.zeros(X.shape)
            dXdt[:,1] = np.linspace(self.freq_range[0][0]*2*np.pi, self.freq_range[0][1]*2*np.pi, X.shape[0])
            dXdt[:,3] = np.linspace(self.freq_range[1][0]*2*np.pi, self.freq_range[1][1]*2*np.pi, X.shape[0])
            return dXdt

        def h(self, X):
            return X[:,0]*np.cos(X[:,1])+X[:,2]*np.cos(X[:,3])

    class Galerkin24(object):   # 2 states (theta1, theta2) 4 base functions [cos(theta1), sin(theta1), cos(theta2), sin(theta2)]
        # Galerkin approximation in finite element method
        def __init__(self):
            # L: int, number of trial (base) functions
            self.L = 4       # trial (base) functions
            
        def psi(self, X):
            trial_functions = [ np.cos(X[:,0]), np.sin(X[:,0]), np.cos(X[:,1]), np.sin(X[:,1])]
            return np.array(trial_functions)
            
        # gradient of trial (base) functions
        def grad_psi(self, X):
            grad_trial_functions = [[           -np.sin(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                    [            np.cos(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                    [ np.zeros(X[:,0].shape[0]),           -np.sin(X[:,1])],\
                                    [ np.zeros(X[:,0].shape[0]),            np.cos(X[:,1])]]
            return np.array(grad_trial_functions)

        # gradient of gradient of trail (base) functions
        def grad_grad_psi(self, X):
            grad_grad_trial_functions = [[[           -np.cos(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                        [ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]],\
                                        [[           -np.sin(X[:,0]), np.zeros(X[:,1].shape[0])],\
                                        [ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])]],\
                                        [[ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                        [ np.zeros(X[:,0].shape[0]),           -np.cos(X[:,1])]],\
                                        [[ np.zeros(X[:,0].shape[0]), np.zeros(X[:,1].shape[0])],\
                                        [ np.zeros(X[:,0].shape[0]),           -np.sin(X[:,1])]]]
            return np.array(grad_grad_trial_functions)

    class Galerkin44(object):   # 4 states (r1, theta1, r2, theta2) 4 base functions [r1*cos(theta1), r1*sin(theta1), r2*cos(theta2), r2*sin(theta2)]
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

    class Galerkin48(object):   # 4 states (r1, theta1, r2, theta2) 8 base functions [r1*cos(theta1), r1*sin(theta1), r2*cos(theta2), r2*sin(theta2), cos(theta1+theta2), sin(theta1+theta2), cos(theta1-theta2), sin(theta1-theta2)]
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

    def plot_innovationa4(signal, filtered_signal, dt):
        r1costheta1 = filtered_signal.X[:,0,:]*np.cos(filtered_signal.X[:,1,:])
        r1sintheta1 = filtered_signal.X[:,0,:]*np.sin(filtered_signal.X[:,1,:])
        h1 = filtered_signal.X[:,0,:]*np.cos(filtered_signal.X[:,1,:])
        h1_hat = np.mean(h1, axis=0)
        h2 = filtered_signal.X[:,2,:]*np.cos(filtered_signal.X[:,3,:])
        h2_hat = np.mean(h2, axis=0)
        a1 = np.mean((h1-h1_hat)*r1costheta1, axis=0)
        a2 = np.mean((h1-h1_hat)*r1sintheta1, axis=0)
        b1 = np.mean((h2-h2_hat)*r1costheta1, axis=0)
        b2 = np.mean((h2-h2_hat)*r1sintheta1, axis=0)

        plt.figure(1)
        plt.plot(signal.t, a1, label='$a_1$')
        plt.plot(signal.t, a2, label='$a_2$')
        plt.plot(signal.t, b1, label='$b_1$')
        plt.plot(signal.t, b2, label='$b_2$')
        plt.legend(fontsize=fontsize-5)

        plt.figure(2)
        for i in range(h2.shape[0]):
            plt.scatter(signal.t, h2[i]-h2_hat, s=0.2)

        plt.figure(3)
        for i in range(h2.shape[0]):
            plt.scatter(signal.t, np.cumsum(a1/2*(h2[i]-h2_hat)*dt), s=0.2)

        plt.figure(4)
        for i in range(h2.shape[0]):
            plt.scatter(signal.t, np.cumsum(a2/2*(h2[i]-h2_hat)*dt), s=0.2)

        plt.figure(5)
        for i in range(h2.shape[0]):
            plt.scatter(signal.t, np.cumsum(b1/2*(h2[i]-h2_hat)*dt), s=0.2)

        plt.figure(6)
        for i in range(h2.shape[0]):
            plt.scatter(signal.t, np.cumsum(b2/2*(h2[i]-h2_hat)*dt), s=0.2)
        return 
    
    Np = 1000

    model=Model2(amp=amp, freq_range=freq_range, min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W)
    galerkin=Galerkin24()
    # model = Model4(amp_range=amp_range, freq_range=freq_range, min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W)
    # galerkin = Galerkin44()
    feedback_particle_filter = FPF(number_of_particles=Np, model=model, galerkin=galerkin, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, particles_ratio=0.01, restrictions=restrictions(model.states),\
                          plot_histogram=True, n_bins = 100, plot_c=True)
    figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
    figs = set_yaxis(figs, model)
    
    # plot_innovationa4(signal, filtered_signal, dt)
    
    plt.show()

    return 

def cfpf():
    from CFPF import CFPF, Figure
    from single_frequency import Model1, Model2, Galerkin, Galerkin1

    def restrictions(states):
        mod = []
        scaling = []
        center = []
        zero = []
        for j in range(len(states)):
            mod.append([])
            scaling.append([])
            center.append([])
            zero.append([])
            for state in states[j]:
                mod[j].append(2*np.pi if 'theta' in state else np.nan)
                scaling[j].append(1/(2*np.pi) if 'omega' in state else 1)
                center[j].append('r' in state)
                zero[j].append('omega' in state)
        return Struct(mod=mod, scaling=scaling, center=center, zero=zero)

    def set_yaxis(figs, models):
        for j in range(len(models)):
            for n, state in enumerate(models[j].states):
                if 'X' in figs.keys():
                    figs['X'].axes[j][n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
                if 'histogram' in figs.keys():
                    figs['histogram'].axes[j][n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
                if 'theta' in state:
                    if 'X' in figs.keys():
                        figs['X'].axes[j][n].set_yticks(np.linspace(0,2*np.pi,5))
                        figs['X'].axes[j][n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                    if 'histogram' in figs.keys():
                        figs['histogram'].axes[j][n].set_yticks(np.linspace(0,2*np.pi,5))
                        figs['histogram'].axes[j][n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                if 'omega' in state:
                    pass
        return figs

    number_of_channels = 2
    Np = [1000, 1000]
    models = []
    galerkins = []
    sigma_B = []
    for j in range(number_of_channels):
        # model = Model1(amp=amp[j], freq_range=freq_range[j], min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W)
        model = Model2(amp_range=amp_range[j], freq_range=freq_range[j], min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W)
        models.append(model)
        galerkins.append(Galerkin1())
        sigma_B.append(signal_type.sigma_B[2*j:2*j+2])
    
    coupled_feedback_particle_filter = CFPF(number_of_channels=number_of_channels, numbers_of_particles=Np, models=models, galerkins=galerkins, sigma_B=sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = coupled_feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, particles_ratio=0.01, restrictions=restrictions([models[j].states for j in range(len(models))]),\
                          plot_histogram=True, n_bins = 100, plot_c=False)
    figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
    figs = set_yaxis(figs, models)

    plt.show()

    return

T = 5
fs = 160
dt = 1/fs
signal_type = Sinusoidals(dt)
signal = Signal(signal_type=signal_type, T=T)
amp = signal_type.amp
amp_range = [[0.5,1.5],[2.5,3.5]]
freq_range = [[1.1,1.3],[3.7,3.9]]
min_sigma_B = 0.01
min_sigma_W = 0.5

if __name__ == "__main__":

    filter_type = sys.argv[1]
    if 'CFPF' in filter_type:
        cfpf()
    elif 'FPF' in filter_type:
        fpf()
    else :
        print('Wrong Filter Type')
    
    