"""
Created on Wed Mar. 27, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.linalg import solve_continuous_are as care
import control

import plotly.offline as pyo
import plotly.graph_objs as go

from Tool import Struct
from System import LTI, BodeDiagram

class Model_LTI(object):
    def __init__(self, A, B, C, D, X0, states, states_label, sigma_V, sigma_W):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.X0 = np.array(X0)
        self.states = states
        self.states_label = states_label
        self.R = np.array(sigma_V) if isinstance(sigma_V[0], list) else np.diag(sigma_V)
        self.Q = np.array(sigma_W) if isinstance(sigma_W[0], list) else np.diag(sigma_W)
        self.d = self.A.shape[0]
        self.n = self.B.shape[1]
        self.m = self.C.shape[0]
    
    def f(self, X, U):
        return np.matmul(self.A, X) + np.matmul(self.B, U)
    
    def h(self, X, U):
        return np.matmul(self.C, X) + np.matmul(self.D, U)

class KF(object):
    def __init__(self, model, dt):
        self.states_label = model.states_label
        self.X = model.X0
        self.f = model.f
        self.h = model.h
        self.dt = dt
        self.P_inf = care(model.A.transpose(), model.C.transpose(), model.Q, model.R)
        self.K_inf = np.matmul(np.matmul(self.P_inf, model.C.transpose()), np.linalg.inv(model.R))
        self.ext_sys = control.ss2tf(
            np.concatenate((np.vstack((model.A,np.matmul(self.K_inf,model.C))),np.vstack((np.zeros(model.A.shape),A-np.matmul(self.K_inf,model.C)))), axis=1),
            np.tile(model.B, (2,1)),
            np.pad(model.C, ((0,0),(model.C.shape[1],0)), 'constant'),
            np.zeros((2*model.C.shape[0], model.B.shape[1]))
        )
    
    def update(self, Y):
        pass
    
    def run(self, Y, show_time=False):
        pass

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
    
    bodeDiagram = BodeDiagram([KF_bode], dB=False)
    bodeDiagram.plot()
  



