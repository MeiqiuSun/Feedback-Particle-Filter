"""
Created on Wed Mar. 20, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct

import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from scipy.io import wavfile

from CFPF import CFPF, Figure
from single_frequency import Model2, Galerkin1
from single_frequency import Model3, Galerkin34

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

if __name__ == "__main__":
    filenames = ['0a7c2a8d_nohash_0']
    # filenames = ['0a7c2a8d_nohash_0','0a9f9af7_nohash_0', '0a9f9af7_nohash_1', '0a9f9af7_nohash_2','0ab3b47d_nohash_0']
    for filename in filenames:
        sampling_rate, samples = wavfile.read("commands/yes/" + filename + '.wav')
        mean = np.mean(samples)
        std = np.std(samples)
        samples = (samples - mean) / std
        samples = samples[6000:8001]
        dt = 1/sampling_rate
        T = (samples.shape[0]-1)*dt
        signal = Struct(t=np.arange(0, T+dt, dt), Y=samples.reshape([1,-1]))

        # plt.plot(signal.t, signal.Y[0], color='gray')
        # plt.axvline(x=signal.t[6000], color='k', linestyle='--')
        # plt.axvline(x=signal.t[8000], color='k', linestyle='--')
        # plt.tick_params(labelsize=20)
        # plt.xlabel('\ntime [s]', fontsize=20)
        
        amp_range = [[4,5],[2,3]]
        freq_range = [[200,250],[500,600]]
        # amp_range = [[2.5,5.0]]
        # freq_range = [[250,300]]
        min_sigma_B = 0.01
        min_sigma_W = 0.1

        number_of_channels = 2
        Np = [1000, 1000]
        models = []
        galerkins = []
        sigma_B = []
        for j in range(number_of_channels):
            model = Model3(amp_range=amp_range[j], freq_range=freq_range[j], min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W)
            models.append(model)
            galerkin = Galerkin34()
            galerkins.append(galerkin)
            sigma_B.append([min_sigma_B,min_sigma_B, min_sigma_B])
            # sigma_B.append([min_sigma_B,min_sigma_B])
        
        cfpf = CFPF(number_of_channels=number_of_channels, numbers_of_particles=Np, models=models, galerkins=galerkins, sigma_B=sigma_B, sigma_W=[min_sigma_W], dt=dt)
        start = time.time()
        filtered_signal = cfpf.run(signal.Y)
        end = time.time()
        print("===========Process Takes Time: {} [s]==============".format(end-start))

        fontsize = 20
        fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, particles_ratio=0.01,\
                              restrictions=restrictions([models[j].states for j in range(len(models))]),\
                              plot_histogram=True, n_bins = 100, plot_c=False)
        print("===========Start Plotting==============") 
        figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
        plt.show()
        
