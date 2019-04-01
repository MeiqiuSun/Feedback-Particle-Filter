"""
Created on Mon Feb. 25, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Signal import Signal, Linear

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    T = 10.
    fs = 160
    dt = 1/fs
    signal_type = Linear(dt)
    signal = Signal(f=signal_type.f, h=signal_type.h, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, X0=signal_type.X0, dt=dt, T=T)

    fontsize = 20
    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    for m in range(signal.Y.shape[0]):
        ax.plot(signal.t, signal.Y[m,:], label='$Y_{}$'.format(m+1))
    plt.legend(fontsize=fontsize-5)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel('time [s]', fontsize=fontsize)
    plt.show()
