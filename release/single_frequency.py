"""
Created on Mon Mar. 11, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF
from Signal import Signal, Sinusoidal

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Figure1(object):  # states (theta1, theta2, ...)
    """Figure: Figures for Feedback Partilce Filter
        Initialize = Figure(fig_property, signal, filtered_signal)
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Members = 
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Methods =
            plot(): plot figures
            plot_signal(axes): plot a figure with observations and filtered observations together, each of axes represents a time series of observation
    """
    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):
        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(9,9))
                ax2 = plt.subplot2grid((4,1),(3,0))
                ax1 = plt.subplot2grid((4,1),(0,0),rowspan=3, sharex=ax2)
                signal_axes = self.plot_signal([ax1, ax2], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_X:
            nrow = 1
            ncol = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow, ncol, figsize=(9*ncol,9), sharex=True)
            X_axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_histogram:
            nrow = 1
            ncol = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow, ncol, figsize=(9*ncol,9), sharex=True)
            histogram_axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\npercentage [%]', fontsize=self.fig_property.fontsize)
        
        
        if self.fig_property.plot_c:
            fig, axes = plt.subplots(self.filtered_signal.c.shape[1], self.filtered_signal.c.shape[0], figsize=(9,7), sharex=True)
            c_axes = self.plot_c(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes, m):
        for i, ax in enumerate(axes):
            if np.mod(i,2)==0:
                ax.plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[m], label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(self.signal.Y)
                ax.set_ylim([minimum, maximum])
            else:
                L2_error = np.sqrt(np.cumsum(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[m])*self.signal.dt)/self.signal.T)/self.signal.sigma_W[m]
                # L2_error = np.sqrt(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[m]))/self.signal.sigma_W[m]
                ax.plot(self.signal.t,L2_error)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(L2_error)
                ax.set_ylim([minimum, maximum])
                ax.set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)
        return axes

    def plot_X(self, axes):
        particles_ratio = 0.01
        Np = self.filtered_signal.X.shape[0]
        sampling_number = np.random.choice(Np, size=int(Np*particles_ratio), replace=False)
        if isiterable(axes):
            for n, ax in enumerate(axes):
                for i in range(sampling_number.shape[0]):
                    state_of_filtered_signal = np.mod(self.filtered_signal.X[sampling_number[i],n,:], 2*np.pi)
                    ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
                minimum, maximum = find_limits(np.mod(self.filtered_signal.X[:,n,:], 2*np.pi))
                ax.set_ylim([minimum, maximum])
                ax.set_yticks(np.linspace(0,2*np.pi,5))
                ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                ax.tick_params(labelsize=self.fig_property.fontsize)
                ax.set_ylabel(r'$\theta_{}$'.format(n+1), fontsize=self.fig_property.fontsize)
        else:
            ax = axes
            for i in range(sampling_number.shape[0]):
                state_of_filtered_signal = np.mod(self.filtered_signal.X[sampling_number[i],0,:], 2*np.pi)
                ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
            minimum, maximum = find_limits(np.mod(self.filtered_signal.X[:,0,:], 2*np.pi))
            ax.set_ylim([minimum, maximum])
            ax.set_yticks(np.linspace(0,2*np.pi,5))
            ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylabel(r'$\theta$', fontsize=self.fig_property.fontsize)
        return axes

    def plot_histogram(self, axes):
        n_bins = 100
        if isiterable(axes):
            for n, ax in enumerate(axes):
                state_of_filtered_signal = np.mod(self.filtered_signal.X[:,n,:], 2*np.pi)
                state_of_signal = np.mod(self.signal.X[n,:], 2*np.pi)
                ax.hist(state_of_filtered_signal[:,-1], bins=n_bins, orientation='horizontal')
                ax.axhline(y=state_of_signal[-1], color='gray', linestyle='--')
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0], symbol=''))
                ax.set_ylim([-0.2*np.pi, 2.2*np.pi])
                ax.set_yticks(np.linspace(0,2*np.pi,5))
                ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                ax.set_ylabel(r'$\theta_{}$ [rad]'.format(n+1), fontsize=self.fig_property.fontsize)
                ax.tick_params(labelsize=self.fig_property.fontsize)
        else:
            ax = axes
            state_of_filtered_signal = np.mod(self.filtered_signal.X[:,0,:], 2*np.pi)
            state_of_signal = np.mod(self.signal.X[0,:], 2*np.pi)
            ax.hist(state_of_filtered_signal[:,-1], bins=n_bins, orientation='horizontal')
            ax.axhline(y=state_of_signal[-1], color='gray', linestyle='--')
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0]))
            ax.set_ylim([-0.2*np.pi, 2.2*np.pi])
            ax.set_yticks(np.linspace(0,2*np.pi,5))
            ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
            ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
            ax.tick_params(labelsize=self.fig_property.fontsize)
        return axes
    
    def plot_c(self, axes):

        if isiterable(axes):
            if self.filtered_signal.c.shape[0]==1:
                for l in range(self.filtered_signal.c.shape[1]):
                    ax = axes[l]
                    ax.plot(self.signal.t, self.filtered_signal.c[0,l,:])
                    ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
                    ax.tick_params(labelsize=self.fig_property.fontsize)
                    minimum, maximum = find_limits(self.filtered_signal.c[0,:,int(self.filtered_signal.c.shape[2]/2):])
                    ax.set_ylim([minimum, maximum])
                    ax.set_ylabel('$c_{}$'.format(l+1), fontsize=self.fig_property.fontsize)
                    if l==0:
                        ax.set_title('$m=1$', fontsize=self.fig_property.fontsize+2)
            else:
                for m in range(self.filtered_signal.c.shape[0]):
                    for l in range(self.filtered_signal.c.shape[1]):
                        ax = axes[l][m]
                        ax.plot(self.signal.t, self.filtered_signal.c[m,l,:])
                        ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
                        ax.tick_params(labelsize=self.fig_property.fontsize)
                        minimum, maximum = find_limits(self.filtered_signal.c[m,:,int(self.filtered_signal.c.shape[2]/2):])
                        ax.set_ylim([minimum, maximum])
                        ax.set_ylabel('$c_{}$'.format(l+1), fontsize=self.fig_property.fontsize)
                    if l==0:
                        ax.set_title('$m={}$'.format(m+1), fontsize=self.fig_property.fontsize+2)
        else:
            ax = axes
            ax.plot(self.signal.t, self.filtered_signal.c[0,0,:])
            ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
            ax.tick_params(labelsize=self.fig_property.fontsize)
            minimum, maximum = find_limits(self.filtered_signal.c[0,0,int(self.filtered_signal.c.shape[2]/2):])
            ax.set_ylim([minimum, maximum])
            ax.set_ylabel('$c$', fontsize=self.fig_property.fontsize)
        return axes

class Figure2(object):  # even states (r1, theta1, r2, theta2, ...)
    """Figure: Figures for Feedback Partilce Filter
        Initialize = Figure(fig_property, signal, filtered_signal)
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Members = 
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Methods =
            plot(): plot figures
            plot_signal(axes): plot a figure with observations and filtered observations together, each of axes represents a time series of observation
    """
    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):
        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(9,9))
                ax2 = plt.subplot2grid((4,1),(3,0))
                ax1 = plt.subplot2grid((4,1),(0,0),rowspan=3, sharex=ax2)
                signal_axes = self.plot_signal([ax1, ax2], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_X:
            nrow = 2
            ncol = int(self.filtered_signal.X.shape[1]/nrow)
            fig, axes = plt.subplots(nrow, ncol, figsize=(9*ncol,9), sharex=True)
            X_axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_histogram:
            nrow = 2
            ncol = int(self.filtered_signal.X.shape[1]/nrow)
            fig, axes = plt.subplots(nrow, ncol, figsize=(9*ncol,9), sharex=True)
            histogram_axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\npercentage [%]', fontsize=self.fig_property.fontsize)
        
        
        if self.fig_property.plot_c:
            fig, axes = plt.subplots(self.filtered_signal.c.shape[1], self.filtered_signal.c.shape[0], figsize=(9,7), sharex=True)
            c_axes = self.plot_c(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes, m):
        for i, ax in enumerate(axes):
            if np.mod(i,2)==0:
                ax.plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[m], label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(self.signal.Y)
                ax.set_ylim([minimum, maximum])
            else:
                L2_error = np.sqrt(np.cumsum(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[m])*self.signal.dt)/self.signal.T)/self.signal.sigma_W[m]
                # L2_error = np.sqrt(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[m]))/self.signal.sigma_W[m]
                ax.plot(self.signal.t,L2_error)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(L2_error)
                ax.set_ylim([minimum, maximum])
                ax.set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)
        return axes

    def plot_X(self, axes):
        particles_ratio = 0.01
        Np = self.filtered_signal.X.shape[0]
        sampling_number = np.random.choice(Np, size=int(Np*particles_ratio), replace=False)
        for n in range(self.filtered_signal.X.shape[1]):
            if self.filtered_signal.X.shape[1]==2:
                ax = axes[np.mod(n,2)]
            else:
                ax = axes[np.mod(n,2)][int(n/2)]
            for i in range(sampling_number.shape[0]):
                if np.mod(n,2)==1:
                    state_of_filtered_signal = np.mod(self.filtered_signal.X[sampling_number[i],n,:], 2*np.pi)
                    minimum, maximum = find_limits(np.mod(self.filtered_signal.X[:,n,:], 2*np.pi))
                    ax.set_ylim([minimum, maximum])
                    ax.set_yticks(np.linspace(0,2*np.pi,5))
                    ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                    if int(n/2)==0:
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                else:
                    state_of_filtered_signal = self.filtered_signal.X[sampling_number[i],n,:]
                    minimum, maximum = find_limits(self.filtered_signal.X[:,n,:])
                    maximum = np.max([maximum,-minimum])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                    if int(n/2)==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
            if np.mod(n,2)==1:
                state_of_signal = np.mod(self.signal.X[n,:], 2*np.pi)
            else:
                state_of_signal = self.signal.X[n,:]
            ax.scatter(self.signal.t, state_of_signal, color='black', s=0.5)
            if np.mod(n,2)==0:
                ax.set_title(r"$r_{},\theta_{}$".format(int(n/2)+1,int(n/2)+1), fontsize=self.fig_property.fontsize+2)
            ax.tick_params(labelsize=self.fig_property.fontsize)
        return axes

    def plot_histogram(self, axes):
        n_bins = 100
        if isiterable(axes):
            for n in range(self.filtered_signal.X.shape[1]):
                if self.filtered_signal.X.shape[1]==2:
                    ax = axes[np.mod(n,2)]
                else:
                    ax = axes[np.mod(n,2)][int(n/2)]
                if np.mod(n,2)==1:
                    state_of_filtered_signal = np.mod(self.filtered_signal.X[:,n,:], 2*np.pi)
                    state_of_signal = np.mod(self.signal.X[n,:], 2*np.pi)
                    ax.set_ylim([-0.2*np.pi,2.2*np.pi])
                    ax.set_yticks(np.linspace(0,2*np.pi,5))
                    ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                    if int(n/2)==0:
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                else:
                    state_of_filtered_signal = self.filtered_signal.X[:,n,:]
                    state_of_signal = self.signal.X[n,:]
                    minimum, maximum = find_limits(state_of_filtered_signal)
                    maximum = np.max([maximum,-minimum])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                    if int(n/2)==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                ax.hist(state_of_filtered_signal[:,-1], bins=n_bins, orientation='horizontal')
                ax.axhline(y=state_of_signal[-1], color='gray', linestyle='--')
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0], symbol=''))
                ax.tick_params(labelsize=self.fig_property.fontsize)
                if np.mod(n,2)==0:
                    ax.set_title(r"$r_{},\theta_{}$".format(int(n/2)+1,int(n/2)+1), fontsize=self.fig_property.fontsize+2)
        else:
            ax = axes
            ax.hist(self.filtered_signal.X[:, 0,-1], bins=n_bins)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0]))
            ax.tick_params(labelsize=self.fig_property.fontsize)

        return axes
    
    def plot_c(self, axes):

        if isiterable(axes):
            if self.filtered_signal.c.shape[0]==1:
                for l in range(self.filtered_signal.c.shape[1]):
                    ax = axes[l]
                    ax.plot(self.signal.t, self.filtered_signal.c[0,l,:])
                    ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
                    ax.tick_params(labelsize=self.fig_property.fontsize)
                    minimum, maximum = find_limits(self.filtered_signal.c[0,:,int(self.filtered_signal.c.shape[2]/2):])
                    ax.set_ylim([minimum, maximum])
                    ax.set_ylabel('$c_{}$'.format(l+1), fontsize=self.fig_property.fontsize)
                    if l==0:
                        ax.set_title('$m=1$', fontsize=self.fig_property.fontsize+2)
            else:
                for m in range(self.filtered_signal.c.shape[0]):
                    for l in range(self.filtered_signal.c.shape[1]):
                        ax = axes[l][m]
                        ax.plot(self.signal.t, self.filtered_signal.c[m,l,:])
                        ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
                        ax.tick_params(labelsize=self.fig_property.fontsize)
                        minimum, maximum = find_limits(self.filtered_signal.c[m,:,int(self.filtered_signal.c.shape[2]/2):])
                        ax.set_ylim([minimum, maximum])
                        ax.set_ylabel('$c_{}$'.format(l+1), fontsize=self.fig_property.fontsize)
                    if l==0:
                        ax.set_title('$m={}$'.format(m+1), fontsize=self.fig_property.fontsize+2)
        else:
            ax = axes
            ax.plot(self.signal.t, self.filtered_signal.c[0,0,:])
            ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='black')
            ax.tick_params(labelsize=self.fig_property.fontsize)
            minimum, maximum = find_limits(self.filtered_signal.c[0,0,int(self.filtered_signal.c.shape[2]/2):])
            ax.set_ylim([minimum, maximum])
            ax.set_ylabel('$c$', fontsize=self.fig_property.fontsize)
        return axes

class Model1(object):   # 1 state (theta)
    def __init__(self, amp, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.N = 1
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

class Model2(object):   # 2 states (r, theta)
    def __init__(self, amp_range, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
        self.N = 2
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

class Galerkin(object):    # 1 states (theta) 2 base functions [cos(theta), sin(theta)]
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

class Galerkin0(object):    # 2 states (r, theta) 2 base functions [r*cos(theta), r*sin(theta)]
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

class Galerkin1(object):    # 2 states (r, theta) 3 base functions [r, r*cos(theta), r*sin(theta)]
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

class Galerkin2(object):    # 2 states (r, theta) 3 base functions [r^2/2, r*cos(theta), r*sin(theta)]
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

class Galerkin3(object):    # 2 states (r, theta) 3 base functions [r^3/6, r*cos(theta), r*sin(theta)]
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

if __name__ == "__main__":

    T = 40
    sampling_rate = 160 # Hz
    dt = 1./sampling_rate
    signal_type = Sinusoidal(dt)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    freq_range = [1.1,1.3]
    # amp_range = [0.5,1.5]
    # model = Model2(amp_range=amp_range, freq_range=freq_range)
    amp = signal_type.amp[0]
    model = Model1(amp=amp, freq_range=freq_range)
    feedback_particle_filter = FPF(number_of_particles=Np, model=model, galerkin=Galerkin(), sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, plot_histogram=True, plot_c=True)
    figure = Figure1(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    plt.show()