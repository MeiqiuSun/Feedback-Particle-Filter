"""
Created on Wed Mar. 27, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
import numpy as np
from scipy.linalg import solve_continuous_are as care
import control
import matplotlib.pyplot as plt

from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio

from Tool import Struct, BodeDiagram

# class KF(object):
#     def __init__(self, model, sigma_B, sigma_W, dt):
#         self.sigma_B = model.modified_sigma_B(np.array(sigma_B))
#         self.sigma_W = model.modified_sigma_W(np.array(sigma_W))
#         self.N = len(self.sigma_B)
#         self.M = len(self.sigma_W)
#         self.dt = dt
#         self.f = model.f
#         self.h = model.h

#     def update(self, y):

def steady_state_sys(A, B, C, R, Q):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    R = np.array(R)
    Q = np.array(Q)
    P_inf = care(A.transpose(), C.transpose(), Q, R)
    K_inf = np.matmul(np.matmul(P_inf, C.transpose()), np.linalg.inv(R))
    A_extend = np.concatenate((np.vstack((A,np.matmul(K_inf,C))),np.vstack((np.zeros(A.shape),A-np.matmul(K_inf,C)))), axis=1)
    B_extend = np.tile(B, (2,1))
    C_extend = np.pad(C, ((0,0),(C.shape[1],0)), 'constant')
    D_extend = np.zeros((C_extend.shape[0], B_extend.shape[1]))
    sys_ss = control.ss2tf(A_extend, B_extend, C_extend, D_extend)
    return sys_ss


if __name__ == "__main__":
    A = [[-1]]
    B = [[1.]]
    C = [[1.]]
    R = [[0.1]]
    Q = [[1]]
    omega = np.logspace(-4, 4, 801)
    sys_ss = steady_state_sys(A, B, C, R, Q)
    dt = 0.01
    T = 10
    t = np.arange(0, T+dt, dt)
    f = 1/(2*np.pi)
    U = np.reshape(np.cos(2.*np.pi*f*t), (-1,t.shape[0]))
    X0=[[1.]]
    t, Y, X = control.forced_response(sys_ss, T=t, U=U, X0=X0)

    mag, phase, omega = control.bode(sys_ss, omega=np.around(omega, decimals=4), dB=True)
    # poles = control.pole(sys_ss)

    KF_bode = BodeDiagram.create_bode_data(name='KF', frequency=omega, magnitude=mag, phase=phase)
    # OFPF_bode = BodeDiagram.load_bode_file('bode.txt')
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
    # OFPF_bode3 = BodeDiagram.load_bode_file('bode3.txt')
    # OFPF_bode21 = BodeDiagram.load_bode_file('bode_2_f_band_noiseless_1.txt')
    # OFPF_bode22 = BodeDiagram.load_bode_file('bode_2_f_band_noiseless_2.txt')
    # OFPF_bode31 = BodeDiagram.load_bode_file('bode_3_f_band_noiseless_1.txt')
    # OFPF_bode32 = BodeDiagram.load_bode_file('bode_3_f_band_noiseless_2.txt')
    # OFPF_bode33 = BodeDiagram.load_bode_file('bode_3_f_band_noiseless_3.txt')
    bodeDiagram = BodeDiagram([KF_bode, OFPF_bode_tail, OFPF_f_fix_noise_bode, OFPF_bode_tail2, OFPF_bode21, OFPF_bode22, OFPF_bode31, OFPF_bode32, OFPF_bode33], dB=False)
    bodeDiagram.plot()
  



