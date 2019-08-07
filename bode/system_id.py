"""
Created on Thu Jul. 18, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from System import LTI

def simulate_output_signal(U):

    pass

def plot_frequency_response():
    from System import BodeDiagram
    OFPF_bode_tail = BodeDiagram.load_bode_file('bode_tail.txt')
    OFPF_bode_tail2 = BodeDiagram.load_bode_file('bode_tail2.txt')
    OFPF_f_band_bode = BodeDiagram.load_bode_file('bode_f_band_noiseless.txt')
    OFPF_f_fix_noise_bode = BodeDiagram.load_bode_file('bode_f_fix_noise.txt')
    OFPF_f_fix_noiseless_bode = BodeDiagram.load_bode_file('bode_f_fix_noiseless.txt')
    OFPF_bode21 = BodeDiagram.load_bode_file('bode21.txt')
    OFPF_bode22 = BodeDiagram.load_bode_file('bode22.txt')
    OFPF_bode31 = BodeDiagram.load_bode_file('bode31.txt')
    OFPF_bode32 = BodeDiagram.load_bode_file('bode32.txt')
    OFPF_bode33 = BodeDiagram.load_bode_file('bode33.txt')
    bodeDiagram = BodeDiagram([OFPF_bode_tail, OFPF_f_fix_noise_bode, OFPF_bode_tail2, OFPF_bode21, OFPF_bode22, OFPF_bode31, OFPF_bode32, OFPF_bode33], dB=False)
    bodeDiagram.plot()
    pass



if __name__ == "__main__":
    
    if sys.argv[1] == 'plot':
        plot_frequency_response()
        pass
    
    from Tool import complex_to_mag_phase
    from Signal import Signal, Sinusoids

    f_min = 0.1 #Hz
    f_max = 10. #Hz
    dt = 0.1/f_max
    T = 10/f_min
    X0 = [[0.]]
    system = LTI(A=[[-1]], B=[[1]], C=[[1]], D=[[0]], sigma_V=[0.01], sigma_W=[0.01], dt=dt)



    if sys.argv[1] == 'KF':
        
        pass


    if sys.argv[1] == 'OFPF':
        from OFPF import OFPF

        number_of_channels = 1
        number_of_frequencies = 101
        omegas = np.zeros((number_of_channels, number_of_frequencies))
        omegas[0,:] = np.logspace(np.log10(2*np.pi*f_min), np.log10(2*np.pi*f_max), number_of_frequencies)

        mag = np.zeros(omegas[0].shape)
        phase = np.zeros(omegas[0].shape)
        for i, omega in enumerate(omegas[0]):
            freq = omega/(2*np.pi)
            U, _ = Signal.create(signal_type=Sinusoids(dt=dt, freq=[freq]), T=T)
            Y, _ = system.run(X0=X0, U=np.reshape(U.value, (-1, U.t.shape[0])))
            signal = Signal(U.t, U.value, Y)
            ofpf = OFPF(number_of_particles=1000, freq_range=[freq, freq], amp_range=[[0.9,1.1],[0.9/(omega**2+1),1.1*omega/(omega**2+1)]],\
                        sigma_amp=[0.01, 0.01], sigma_W=[0.1, 0.1], dt=dt, sigma_freq=0.1, tolerance=0)
            ofpf.run(signal.value, show_time=True)
            mag[i], phase[i] = complex_to_mag_phase(ofpf.get_gain())
            print('omega = {}[rad/s], mag = {}, phase = {}[rad]'.format(omega, mag[i], phase[i]))
        
        np.savetxt('bode.txt', np.transpose(np.array([omegas[0], mag, phase])), fmt='%.4f')
        pass

    if sys.argv[1] == 'COFPF':
        from COFPF import COFPF

        number_of_channels = 2
        number_of_frequencies = int(100/number_of_channels)+1
        omega_difference = (np.log10(2*np.pi*f_max)-np.log10(2*np.pi*f_min))/number_of_channels
        omegas = np.zeros((number_of_channels, number_of_frequencies))
        for nc in range(number_of_channels):            
            omegas[nc,:] = np.logspace(np.log10(2*np.pi*f_min)+nc*omega_difference, np.log10(2*np.pi*f_min)+(nc+1)*omega_difference, number_of_frequencies)

        
        pass

    pass