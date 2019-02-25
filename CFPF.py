"""
Created on Wed Feb. 13, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from FPF import FPF, h
from Signal import Signal
from Tool import Struct, isiterable

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

    def __init__(self, number_of_channels, number_of_particles, f_min, f_max, sigma_W, dt, h):
        self.fpf = []
        self.N = number_of_channels
        for n in range(self.N):
            self.fpf.append(FPF(number_of_particles=number_of_particles[n], f_min=f_min[n], f_max=f_max[n], sigma_W=sigma_W, dt=dt, h=h, indep_amp_update=False))
        self.h_hat = np.zeros(len(sigma_W))
        return

    def calculate_h_hat(self):
        """Summation of h_hat for all FPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for n in range(self.N):        
            h_hat += self.fpf[n].h_hat
        return h_hat
    
    def update(self, y):
        """Update each FPF channel with new observation data y"""
        for n in range(self.N):
            self.fpf[n].update(y-self.h_hat+self.fpf[n].h_hat)
        self.h_hat = self.calculate_h_hat()
        return
    
    def run(self, y):
        """Run CFPF with time series observations(y)"""
        h_hat = np.zeros(y.shape)
        h_hat_n = []
        theta = []
        amp = []
        freq = []
        for n in range(self.N):
            h_hat_n.append(np.zeros(y.shape))
            theta.append(np.zeros([self.fpf[n].particles.Np, y.shape[1]]))
            freq.append(np.zeros([self.fpf[n].particles.Np, y.shape[1]]))
            amp.append(np.zeros([self.fpf[n].particles.Np, y.shape[1]]))

        for k in range(y.shape[1]):
            self.update(y[:,k])
            h_hat[:,k] = self.h_hat
            for n in range(self.N):
                h_hat_n[n][:,k] = self.fpf[n].h_hat
                theta[n][:,k] = self.fpf[n].particles.get_theta()
                freq[n][:,k] = self.fpf[n].particles.get_freq()
                amp[n][:,k] = self.fpf[n].particles.get_amp()
        return Struct(h_hat=h_hat, h_hat_n=h_hat_n, theta=theta, freq=freq, amp=amp)

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
            fig, axes = plt.subplots(self.signal.Y.shape[0]+1, 1, figsize=(9,7), sharex=True)
            signal_axes = self.plot_signal(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)
        
        if self.fig_property.plot_theta:
            fig, axes = plt.subplots(len(self.filtered_signal.theta),1, figsize=(9,7), sharex=True)
            theta_axes = self.plot_theta(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)
            plt.title('Particles', fontsize=self.fig_property.fontsize+2)
        
        if self.fig_property.plot_freq:
            fig, axes = plt.subplots(len(self.filtered_signal.freq),1, figsize=(9,7), sharex=True)
            freq_axes = self.plot_freq(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)
            plt.title('Frequency of Particles', fontsize=self.fig_property.fontsize+2)
        
        if self.fig_property.plot_amp:
            fig, axes = plt.subplots(len(self.filtered_signal.amp),1, figsize=(9,7), sharex=True)
            amp_axes = self.plot_amp(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)
            plt.title('Amplitude of Particles', fontsize=self.fig_property.fontsize+2)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes):
        for i, ax in enumerate(axes[0:-1]):
            ax.plot(self.signal.t, self.signal.Y[i], label=r'signal')
            ax.plot(self.signal.t, self.filtered_signal.h_hat[i,:], label=r'estimation')
            ax.legend(fontsize=self.fig_property.fontsize-5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
        ax = axes[-1]
        for n in range(len(self.filtered_signal.h_hat_n)):
            ax.plot(self.signal.t, self.filtered_signal.h_hat_n[n][0,:], label="$FPF_{}$".format(n+1))
            ax.legend(fontsize=self.fig_property.fontsize-5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
        return axes

    def plot_theta(self, axes):
        if isiterable(axes):
            for n, ax in enumerate(axes):
                for k in range(self.filtered_signal.theta[n].shape[0]):
                    ax.scatter(self.signal.t, self.filtered_signal.theta[n][k,:], s=0.5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                ax.set_ylabel(r'$\theta_{}$ [rad]'.format(n+1), fontsize=self.fig_property.fontsize)
        else:
            ax = axes
            for k in range(self.filtered_signal.theta[0].shape[0]):
                ax.scatter(self.signal.t, self.filtered_signal.theta[0][k,:], s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
            
        return axes
    
    def plot_freq(self, axes):
        if isiterable(axes):
            for n, ax in enumerate(axes):
                for k in range(self.filtered_signal.freq[n].shape[0]):
                    ax.scatter(self.signal.t, self.filtered_signal.freq[n][k,:], s=0.5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                ax.set_ylabel('$\omega_{}$ [Hz]'.format(n+1), fontsize=self.fig_property.fontsize)
        else:
            ax = axes
            for k in range(self.filtered_signal.freq[0].shape[0]):
                ax.scatter(self.signal.t, self.filtered_signal.freq[0][k,:], s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylabel('$\omega$ [Hz]', fontsize=self.fig_property.fontsize)
            
        return axes
    
    def plot_amp(self, axes):
        if isiterable(axes):
            for n, ax in enumerate(axes):
                for k in range(self.filtered_signal.amp[n].shape[0]):
                    ax.scatter(self.signal.t, self.filtered_signal.amp[n][k,:], s=0.5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                ax.set_ylabel('$A_{}$'.format(n+1), fontsize=self.fig_property.fontsize)
        else:
            ax = axes
            for k in range(self.filtered_signal.amp[0].shape[0]):
                ax.scatter(self.signal.t, self.filtered_signal.amp[0][k,:], s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylabel('$A$', fontsize=self.fig_property.fontsize)
        
        return axes
    
if __name__ == "__main__":
    # N states of frequency inputs
    freq = [340, 720]
    # M-by-N amplitude matrix
    amp = [[1,0.8]]
    # N states of state noises
    sigma_B = [0.0001,0.0001]
    # M states of signal noises
    sigma_W = [0.0001]

    T = 1
    sampling_rate = 16000 # Hz
    dt = 1./sampling_rate
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    
    number_of_channels = 2
    N = [100, 100]
    f_max = [400, 800]
    f_min = [280, 640]
    coupled_feedback_particle_filter = CFPF(number_of_channels=number_of_channels, number_of_particles=N, f_min=f_min, f_max=f_max, sigma_W=sigma_W, dt=dt, h=h)
    filtered_signal = coupled_feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=True, plot_signal=True, plot_theta=True, plot_freq=True, plot_amp=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_matrix(ax, feedback_particle_filter.particles.update_sync_matrix())
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_particle(ax, feedback_particle_filter.particles.check_sync())

    # plt.show()