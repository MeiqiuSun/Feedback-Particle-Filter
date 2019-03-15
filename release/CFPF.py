"""
Created on Wed Feb. 13, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF
from Signal import Signal, Sinusoidals
from single_frequency import Model2, Galerkin1

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class CFPF(object):
    """CFPF: Coupled Feedback Partilce Filter
        Notation Note:
            M: number of observations(y)
        Initialize = CFPF(number_of_channels, number_of_particles, f_min, f_max, sigma_W, dt, h)
            number_of_channels: int, number of FPF channels
            number_of_particles: 1D list of int with length N, a list of number of particles for each channel
            f_min: 1D list of float with length N, a list of minimum frequencies in Hz for each channel
            f_max: 1D list of float with length N, a list of maximum frequencies in Hz for each channel
            sigma_W: 1D list of float with legnth M, standard deviations of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
        Members = 
            N: number of channels
            fpf: 1D list of FPF with length N, a list of N channels of FPF
            h_hat: numpy array with the shape of (M,), filtered observations
        Methods =
            calculate_h_hat(): Summation of h_hat for all FPF channels
            update(y): Update each FPF channel with new observation data y
            run(y): Run CFPF with time series observations(y)
    """

    def __init__(self, number_of_channels, numbers_of_particles, models, galerkins, sigma_B, sigma_W, dt):
        self.fpf = []
        self.Nc = number_of_channels
        self.dt = dt
        for j in range(self.Nc):
            self.fpf.append(FPF(number_of_particles=numbers_of_particles[j], model=models[j], galerkin=galerkins[j], sigma_B=sigma_B[j], sigma_W=sigma_W, dt=self.dt))
        self.h_hat = np.zeros(len(sigma_W))
        return

    def calculate_h_hat(self):
        """Summation of h_hat for all FPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for j in range(self.Nc):        
            h_hat += self.fpf[j].h_hat
        return h_hat
    
    def update(self, y):
        """Update each FPF channel with new observation data y"""
        for j in range(self.Nc):
            self.fpf[j].update(y-self.h_hat+self.fpf[j].h_hat)
        self.h_hat = self.calculate_h_hat()
        return
    
    def run(self, y):
        """Run CFPF with time series observations(y)"""
        h_hat = np.zeros([self.Nc+1, y.shape[0], y.shape[1]])
        h = []
        X = []

        for j in range(self.Nc):
            h.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].M, y.shape[1]]))
            X.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].N, y.shape[1]]))

        for k in range(y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(y[:,k])
            h_hat[0,:,k] = self.h_hat
            for j in range(self.Nc):
                h_hat[j+1,:,k] = self.fpf[j].h_hat
                h[j][:,:,k] = self.fpf[j].particles.h
                X[j][:,:,k] = self.fpf[j].particles.X
                
        return Struct(h_hat=h_hat, h=h, X=X)

class Figure(object):
    """Figure: Figures for Coupled Feedback Partilce Filter
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
            plot_signal(axes): plot a figure with observations and filtered observations together,
                                each of axes represents a time series of observation. 
                                The last ax plot all channels of h_hat_n.
            plot_theta(ax): plot theta for each particles
            plot_freq(ax): plot frequency for each particles
            plot_amp(ax): plot amplitude for each particles
    """

    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):
        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(9,9))
                ax3 = plt.subplot2grid((5,1),(4,0))
                ax2 = plt.subplot2grid((5,1),(3,0), sharex=ax3)
                ax1 = plt.subplot2grid((5,1),(0,0),rowspan=3, sharex=ax3)
                signal_axes = self.plot_signal([ax1, ax2, ax3], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
        
        if self.fig_property.plot_X:
            ncol = len(self.filtered_signal.X)
            fig = plt.figure(figsize=(9*ncol,9))
            axes = []
            for j in range(ncol):
                nrow = self.filtered_signal.X[j].shape[1]
                axes.append([])
                for n in range(nrow):
                    if n==0:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j)))
                    else:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j),sharex=axes[j][0]))
                    if not n==(nrow-1):
                        plt.setp(axes[j][n].get_xticklabels(), visible=False)
            X_axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_histogram:
            ncol = len(self.filtered_signal.X)
            fig = plt.figure(figsize=(9*ncol,9))
            axes = []
            for j in range(ncol):
                nrow = self.filtered_signal.X[j].shape[1]
                axes.append([])
                for n in range(nrow):
                    if n==0:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j)))
                    else:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j),sharex=axes[j][0]))
                    if not n==(nrow-1):
                        plt.setp(axes[j][n].get_xticklabels(), visible=False)
            histogram_axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\npercentage [%]', fontsize=self.fig_property.fontsize)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes, m):
        axes[0].plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal')
        axes[0].plot(self.signal.t, self.filtered_signal.h_hat[0,m], label=r'estimation')
        axes[0].legend(fontsize=self.fig_property.fontsize-5)
        axes[0].tick_params(labelsize=self.fig_property.fontsize)
        minimum, maximum = find_limits(self.signal.Y)
        axes[0].set_ylim([minimum, maximum])

        L2_error = np.sqrt(np.cumsum(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[0,m])*self.signal.dt)/self.signal.T)/self.signal.sigma_W[m]
        # L2_error = np.sqrt(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[0,m]))/self.signal.sigma_W[m]
        axes[1].plot(self.signal.t,L2_error)
        axes[1].tick_params(labelsize=self.fig_property.fontsize)
        minimum, maximum = find_limits(L2_error)
        axes[1].set_ylim([minimum, maximum])
        axes[1].set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)

        for j in range(self.filtered_signal.h_hat.shape[0]-1):
            for i in range(self.filtered_signal.h[j].shape[0]):
                axes[2].scatter(self.signal.t, self.filtered_signal.h[j][i,m,:], color='C{}'.format(j+1), alpha=0.01)
        for j in range(self.filtered_signal.h_hat.shape[0]-1):
            axes[2].plot(self.signal.t, self.filtered_signal.h_hat[j+1,m,:], color='C{}'.format(j+1), label=r'$_{}\hat h$'.format(j+1))
        axes[2].tick_params(labelsize=self.fig_property.fontsize)
        minimum, maximum = find_limits(self.filtered_signal.h_hat[1:,m])
        axes[2].set_ylim([minimum, maximum])
        axes[2].legend(fontsize=self.fig_property.fontsize-5)
        return axes
    
    def plot_X(self, axes):
        particles_ratio = 0.01
        n_signal = 0
        for j in range(len(axes)):
            Np = self.filtered_signal.X[j].shape[0]
            sampling_number = np.random.choice(Np, size=int(Np*particles_ratio), replace=False)
            for n in range(self.filtered_signal.X[j].shape[1]):
                ax = axes[j][n]
                for i in range(sampling_number.shape[0]):
                    if n==1:
                        state_of_filtered_signal = np.mod(self.filtered_signal.X[j][sampling_number[i],n,:], 2*np.pi)
                    else:
                        state_of_filtered_signal = self.filtered_signal.X[j][sampling_number[i],n,:]
                    ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
                if n==1:
                    ax.set_ylim([-0.2*np.pi, 2.2*np.pi])
                    ax.set_yticks(np.linspace(0,2*np.pi,5))
                    ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                    if j==0:
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                    state_of_signal = np.mod(self.signal.X[n_signal,:], 2*np.pi)
                else:
                    filtered_signal_limits = find_limits(self.filtered_signal.X[j][:,n,:])
                    if j==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                    state_of_signal = self.signal.X[n_signal,:]
                    signal_limits = find_limits(self.signal.X[n_signal])
                    maximum = np.max([-filtered_signal_limits[0], filtered_signal_limits[1], -signal_limits[0], signal_limits[1]])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                ax.scatter(self.signal.t, state_of_signal, color='black', s=0.5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                if n==0:
                    ax.set_title('channel {}'.format(j+1), fontsize=self.fig_property.fontsize+2)
                n_signal += 1
        return axes
    
    def plot_histogram(self, axes):
        n_bins = 100
        n_signal = 0
        max_percentage = 0
        for j in range(len(axes)):
            for n in range(self.filtered_signal.X[j].shape[1]):
                ax = axes[j][n]
                if n==1:
                    state_of_filtered_signal = np.mod(self.filtered_signal.X[j][:,n,:], 2*np.pi)
                    state_of_signal = np.mod(self.signal.X[n_signal,:], 2*np.pi)
                else:
                    state_of_filtered_signal = self.filtered_signal.X[j][:,n,:]
                    state_of_signal = self.signal.X[n_signal,:]
                ax.hist(state_of_filtered_signal[:,-1], bins=n_bins, orientation='horizontal')
                ax.axhline(y=state_of_signal[-1], color='gray', linestyle='--')
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X[j].shape[0], symbol=''))
                _, maximum = ax.get_xlim()
                max_percentage = np.max([maximum, max_percentage])
                ax.tick_params(labelsize=self.fig_property.fontsize)
                if n==1:
                    ax.set_ylim([-0.2*np.pi, 2.2*np.pi])
                    ax.set_yticks(np.linspace(0,2*np.pi,5))
                    ax.set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                    if j==0:
                        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
                    state_of_signal = np.mod(self.signal.X[n_signal,:], 2*np.pi)
                else:
                    filtered_signal_limits = find_limits(self.filtered_signal.X[j][:,n,:])
                    if j==0:
                        ax.set_ylabel('$r$    ', fontsize=self.fig_property.fontsize, rotation='horizontal')
                    state_of_signal = self.signal.X[n_signal,:]
                    signal_limits = find_limits(self.signal.X[n_signal])
                    maximum = np.max([-filtered_signal_limits[0], filtered_signal_limits[1], -signal_limits[0], signal_limits[1]])
                    minimum = -maximum
                    ax.set_ylim([minimum, maximum])
                if n==0:
                    ax.set_title('channel {}'.format(j+1), fontsize=self.fig_property.fontsize+2)
                n_signal += 1
        for j in range(len(axes)):
            for _, ax in enumerate(axes[j]):
                ax.set_xlim([0, max_percentage])
        return axes

if __name__ == "__main__":
    
    T = 5
    fs = 160
    dt = 1/fs
    signal_type = Sinusoidals(dt)
    signal = Signal(signal_type=signal_type, T=T)

    number_of_channels = 2
    Np = [1000, 1000]
    amp_range = [[0.5,1.5],[2.5,3.5]]
    freq_range = [[1.1,1.3],[3.7,3.9]]
    models = []
    galerkins = []
    sigma_B = []
    min_sigma_B = 0.01
    min_sigma_W = 0.5
    for j in range(number_of_channels):
        models.append(Model2(amp_range=amp_range[j], freq_range=freq_range[j], min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W))
        galerkins.append(Galerkin1())
        sigma_B.append(signal_type.sigma_B[2*j:2*j+2])
    
    coupled_feedback_particle_filter = CFPF(number_of_channels=number_of_channels, numbers_of_particles=Np, models=models, galerkins=galerkins, sigma_B=sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = coupled_feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, plot_histogram=True, plot_c=False)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    plt.show()