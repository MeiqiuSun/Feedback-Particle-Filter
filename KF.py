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

from Tool import Struct

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

def bode_data(omega, mag, phase):
    return Struct(omega=omega, mag=mag, phase=phase)

def bode_plot(*args):
    data =[None] * (2*len(args))
    for i, arg in enumerate(args):
        mag_trace = go.Scatter(
            x = arg.omega,
            y = arg.mag,
            yaxis = 'y1'
        )

        phase_trace = go.Scatter(
            x = arg.omega,
            y = arg.phase,
            yaxis = 'y2'
        )
        data[2*i] = mag_trace
        data[2*i+1] = phase_trace

    layout = go.Layout(
        title = 'Bode Diagram',
        xaxis=dict(
            title = 'Frequency [rad/s]',
            type = 'log'
        ),
        yaxis=dict(
            title = 'Magnitude',
            domain = [0.5,1],
            anchor = 'y1'
        ),
        yaxis2=dict(
            title = 'Phase [rad]',
            range = [-np.pi-0.1, 0.1],
            tick0 = -np.pi,
            dtick = np.pi/4,
            domain = [0,0.5],
            anchor = 'y2'
        )
    )
    # fig = tools.make_subplots(rows=2, cols=1)
    # fig.append_trace(mag_trace, 1, 1)
    # fig.append_trace(phase_trace, 2, 1)
    # fig['layout'].update(height=600, width=600, title='Stacked subplots')
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename='bode_diagram')
    return fig

if __name__ == "__main__":
    A = [[-1]]
    B = [[1.]]
    C = [[1.]]
    R = [[0.1]]
    Q = [[1]]
    omega = np.logspace(-4, 4, 1000)
    sys_ss = steady_state_sys(A, B, C, R, Q)
    dt = 0.01
    T = 10
    t = np.arange(0, T+dt, dt)
    f = 1/(2*np.pi)
    U = np.reshape(np.cos(2.*np.pi*f*t), (-1,t.shape[0]))
    X0=[[1.]]
    t, Y, X = control.forced_response(sys_ss, T=t, U=U, X0=X0)

    mag, phase, omega = control.bode(sys_ss, omega=omega, dB=True)
    poles = control.pole(sys_ss)
    KF_bode = bode_data(omega, mag, phase)
    omega, mag, phase  = np.loadtxt('bode.txt', unpack=True)
    OFPF_bode = bode_data(omega, mag, phase)

    fig = bode_plot(KF_bode, OFPF_bode)
    plt.figure()
    plt.plot(t, U[0,:])
    plt.plot(t, Y)

    # sys_ss = control.ss2tf(A, B, C, np.array([[0]]))
    # poles = control.pole(sys_ss)
    # mag, phase, omega = control.bode(sys_ss, omega=omega, dB=True, deg=True)
    # control.root_locus(sys_ss)
    # fig = bode_plot(mag, phase, omega)
    plt.show()




