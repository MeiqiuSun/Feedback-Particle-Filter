"""
Created on Wed Apr. 17, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from COFPF import COFPF, Figure
from Signal import Signal, Sinusoidals

import sympy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class COFPO(object):
    """COFPO: Coupled Oscillating Feedback Particle Observer
        Notation Note:
            N: int, number of states(X)
            M: int, number of observations(Y)
        Initialize = COFPO(number_of_channels, number_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt)
            number_of_channels: int, number of FPF channels
            number_of_particles: 1D list of int with length Nc, a list of number of particles for each channel
            amp_ranges: 2D list of float with length M-by-2, a list of amplitude range [amp_min, amp_max] for each observation (Ym)
            freq_ranges: 2D list of float with length Nc-by-2, a list of frequency range [f_min, f_max] for each channel
            sigma_B: 1D list of float with legnth N, standard deviations of process noise (same for each channel)
            sigma_W: 1D list of float with legnth M, standard deviations of noise of observations(Y)
            dt: float, size of time step in sec
            known_observations: 1D list of int with length no greater than M, number of known observations in the testing stage (start from 1 to M) e.g. [1,4,9] in M=9 case
            min_sigma_W: int, smallest value of sigma_W
        Members = 
            cofpf_train: 
            r: numpy array with the shape of (M,Nc)
            cofpf_test:
        Methods =
            update(Y): Update each FPF channel with new observation data y
            run(Y): Run CFPF with time series observations(y)
    """

    def __init__(self, number_of_channels, numbers_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt, known_observations, min_sigma_W = 0.1):
        self.cofpf_train = COFPF(number_of_channels, numbers_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt, min_sigma_W)
        self.known_observations = np.array(known_observations)
        self.unknown_observations = np.delete(np.arange(len(amp_ranges))+1, self.known_observations-1)
        self.r = np.zeros([len(amp_ranges), number_of_channels])
        amp_ranges = [amp_ranges[m-1] for m in self.known_observations]
        sigma_W = [sigma_W[m-1] for m in self.known_observations]
        known_observations.insert(0,0)
        sigma_B = [sigma_B[m] for m in known_observations]
        known_observations.pop(0)
        self.cofpf_test = COFPF(number_of_channels, numbers_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt, min_sigma_W)
        return

    def train(self, Y):
        print("Starting Trainning ...")
        self.cofpf_train.run(Y)
        print("End Trainning ...")
        for j in range(self.cofpf_train.Nc):
            self.r[:,j] = np.mean(self.cofpf_train.fpf[j].particles.X[:,1:], axis=0)
        print(self.r)

    def test(self, Y):
        filtered_signal = self.cofpf_test.run(Y)
        Y_hat = np.zeros([self.r.shape[0], Y.shape[1]])
        for m_index, m in enumerate(self.known_observations):
            Y_hat[m-1,:] = filtered_signal.h_hat[0,m_index,:]
        for m in self.unknown_observations:
            for nc in range(self.cofpf_test.Nc):
                Y_hat[m-1,:] += self.r[m-1,nc]*np.mean(np.cos(filtered_signal.X[nc][:,0,:]), axis=0)
        fontsize = 20
        fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, particles_ratio=0.01,\
                            plot_histogram=True, n_bins = 100, plot_c=False)
        figs = Figure(fig_property=fig_property, signal=signal_test, filtered_signal=filtered_signal).plot()
        return Y_hat

        
        


if __name__ == "__main__":
    T = 50.
    fs = 160
    dt = 1/fs
    signal_type1 = Sinusoidals(dt, amp=[1,3], freq=[1.2,3.8])
    signal_type2 = Sinusoidals(dt, amp=[2], freq=[3.8])
    
    signal_train = Signal(signal_type1, T) + Signal(signal_type2, T)
    signal_test = Signal(signal_type1, T)
    
    cofpo = COFPO(number_of_channels=2, numbers_of_particles=[1000,1000], amp_ranges=[[0,4],[0,4]], freq_ranges=[[1,2],[3,4]], sigma_B=[0, 1, 1], sigma_W=[1,1], dt=signal_train.dt, known_observations=[1])
    cofpo.train(signal_train.Y)
    Y_hat = cofpo.test(signal_train.Y[0:1,:])
    plt.figure()
    plt.plot(signal_train.t, signal_train.Y[0])
    plt.plot(signal_train.t, Y_hat[0])
    plt.figure()
    plt.plot(signal_train.t, signal_train.Y[1])
    plt.plot(signal_train.t, Y_hat[1])
    plt.show()