"""
Created on Mon Mar. 11, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from FPF import FPF
from Signal import Signal, Sinusoidal
from Tool import Struct, isiterable, find_limits

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class Figure(object):
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
            plot_theta(ax): plot theta for each particles
            plot_freq(ax): plot frequency for each particles and the average frequency
            plot_amp(ax): plot amplitude for each particles and the average amplitude
            plot_sync_matrix(ax, sync_matrix): plot synchronized matrix
            plot_sync_particles(ax, sync_particles): plot synchronized particles
    """
    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):
        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(9,9))
                # print(fig)
                ax2 = plt.subplot2grid((4,1),(3,0))
                ax1 = plt.subplot2grid((4,1),(0,0),rowspan=3, sharex=ax2)
                signal_axes = self.plot_signal([ax1, ax2], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                # fig = plt.gcf()
                # fig.figsize=(9,9)
                # fig.sharex = True
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
                L2_error = np.sqrt(np.cumsum(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[m])*self.signal.dt))
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
                    if int(n/2)==0:
                        ax.set_yticklabels(['$0$','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                    else:
                        ax.set_yticklabels('')
                else:
                    state_of_filtered_signal = self.filtered_signal.X[sampling_number[i],n,:]
                    minimum, maximum = find_limits(self.filtered_signal.X[:,n,:])
                    maximum = np.max([maximum,-minimum])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                    if int(n/2)==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                    else:
                        ax.set_ylabel('')
                ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
            if np.mod(n,2)==1:
                state_of_signal = np.mod(self.signal.X[n,:], 2*np.pi)
            else:
                state_of_signal = self.signal.X[n,:]
            ax.scatter(self.signal.t, state_of_signal, color='black', s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
        # ax.legend(fontsize=self.fig_property.fontsize-5)
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
                    minimum, maximum = find_limits(state_of_filtered_signal)
                    ax.set_ylim([minimum, maximum])
                    ax.set_yticks(np.linspace(0,2*np.pi,5))
                    if int(n/2)==0:
                        ax.set_yticklabels(['$0$','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'])
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                    else:
                        ax.set_yticklabels('')
                else:
                    state_of_filtered_signal = self.filtered_signal.X[:,n,:]
                    state_of_signal = self.signal.X[n,:]
                    minimum, maximum = find_limits(state_of_filtered_signal)
                    maximum = np.max([maximum,-minimum])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                    if int(n/2)==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                    else:
                        ax.set_ylabel('')
                ax.hist(state_of_filtered_signal[:,-1], bins=n_bins, orientation='horizontal')
                ax.axhline(y=state_of_signal[-1], color='gray', linestyle='--')
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0], symbol=''))
                ax.tick_params(labelsize=self.fig_property.fontsize)
        else:
            ax = axes
            ax.hist(self.filtered_signal.X[:, 0,-1], bins=n_bins)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0]))
            ax.tick_params(labelsize=self.fig_property.fontsize)

        return axes

class Model(object):
    def __init__(self):
        self.N = 2
        self.M = 1
        self.X0_range = [[1,2],[0,2*np.pi]]
        self.min_sigma_B = 0.01
        self.min_sigma_W = 0.1

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
        freq_min=0.9
        freq_max=1.1
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = np.linspace(freq_min*2*np.pi, freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1]) 

class Galerkin(object):    # 2 states (r, theta) 2 base functions [r*cos(theta), r*sin(theta)]
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
        
if __name__ == "__main__":

    T = 10
    sampling_rate = 160 # Hz
    dt = 1./sampling_rate
    signal_type = Sinusoidal(dt)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    feedback_particle_filter = FPF(number_of_particles=Np, model=Model(), galerkin=Galerkin(), signal_type=signal_type)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, plot_histogram=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    plt.show()