"""
Created on Wed Feb. 13, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF
from Signal import Signal, Sinusoidals
from single_frequency import Model, Galerkin

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
        X = []

        for j in range(self.Nc):
            X.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].N, y.shape[1]]))

        for k in range(y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(y[:,k])
            h_hat[-1,:,k] = self.h_hat
            for j in range(self.Nc):
                h_hat[j,:,k] = self.fpf[j].h_hat
                X[j][:,:,k] = self.fpf[j].particles.X
                
        return Struct(h_hat=h_hat, X=X)

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
                ax2 = plt.subplot2grid((4,1),(3,0))
                ax1 = plt.subplot2grid((4,1),(0,0),rowspan=3, sharex=ax2)
                signal_axes = self.plot_signal([ax1, ax2], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

            # fig, axes = plt.subplots(self.signal.Y.shape[0]+1, 1, figsize=(9,7), sharex=True)
            # signal_axes = self.plot_signal(axes)
            # fig.add_subplot(111, frameon=False)
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes, m):
        for i, ax in enumerate(axes):
            if np.mod(i,2)==0:
                ax.plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[-1,m], label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(self.signal.Y)
                ax.set_ylim([minimum, maximum])
            else:
                L2_error = np.sqrt(np.cumsum(np.square(self.signal.Y[m]-self.filtered_signal.h_hat[-1,m])*self.signal.dt))
                ax.plot(self.signal.t,L2_error)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                minimum, maximum = find_limits(L2_error)
                ax.set_ylim([minimum, maximum])
                ax.set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)
        return axes
        
   
if __name__ == "__main__":
    
    T = 10
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
    for j in range(number_of_channels):
        models.append(Model(amp_range=amp_range[j], freq_range=freq_range[j]))
        galerkins.append(Galerkin())
        sigma_B.append(signal_type.sigma_B[2*j:2*j+2])
    
    coupled_feedback_particle_filter = CFPF(number_of_channels=number_of_channels, numbers_of_particles=Np, models=models, galerkins=galerkins, sigma_B=sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = coupled_feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=False, plot_histogram=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    plt.show()