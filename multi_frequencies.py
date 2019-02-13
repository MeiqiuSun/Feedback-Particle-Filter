"""
Created on Thu Jan. 31, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
from FPF import FPF, Signal

def h(amp, x):
        return amp*np.cos(x)

if __name__ == "__main__":
    # N states of frequency inputs
    freq = [0.65, 1.15]
    # M-by-N amplitude matrix
    amp = [[1,1]]
    # N states of state noises
    sigma_B = [0.1,0.1]
    # M states of signal noises
    sigma_W = [0.1]

    T = 10.
    dt = 0.01
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    h_hat = np.zeros(signal.Y.shape)
    
    N=100
    feedback_particle_filter = FPF(number_of_particles=N, f_min=0.6, f_max=1.2, sigma_W=sigma_W, dt=dt, h=h)
    theta = np.zeros([N, signal.Y.shape[1]])
    freq_hat = np.zeros([N, signal.Y.shape[1]])

    for k in range(signal.Y.shape[1]):
        feedback_particle_filter.update(signal.Y[:,k])
        h_hat[:,k] = feedback_particle_filter.h_hat
        theta[:,k] = feedback_particle_filter.particles.get_theta()
        freq_hat[:,k] = feedback_particle_filter.particles.get_freq()
    feedback_particle_filter.particles.update_sync_matrix()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(signal.Y.shape[0],1, figsize=(9,7))
    try:
        iterator = iter(axes)
    except TypeError:
        ax = axes
        ax.plot(signal.t, signal.Y[0])
        ax.plot(signal.t, h_hat[0,:])
    else:
        for i, ax in enumerate(axes):
            ax.plot(signal.t, signal.Y[i])
            ax.plot(signal.t, h_hat[i,:])
    
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.scatter(signal.t, theta[k,:], s=0.5)
    
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    for k in range(N):
        ax.plot(signal.t, freq_hat[k,:]) 

    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax.pcolor(range(N), range(N), feedback_particle_filter.particles.update_sync_matrix(), cmap='gray', vmin=0, vmax=1)
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax.plot(feedback_particle_filter.particles.check_sync())

    plt.show()