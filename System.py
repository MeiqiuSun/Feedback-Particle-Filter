"""
Created on Thu Jul. 18, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

from Tool import Struct, default_colors
from Particles import Particles

class LTI(object):
    def __init__(self, A, B, C, D, sigma_V, sigma_W, dt):
        
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)

        self.sigma_V = np.array(sigma_V)
        self.sigma_W = np.array(sigma_W)
        self.dt = dt

    def states_constraints(self, X):
        return X
    
    def f(self, X, U):
        return np.matmul(self.A, X) + np.matmul(self.B, U)
    
    def h(self, X, U):
        return np.matmul(self.C, X) + np.matmul(self.D, U)

    def update(self, Xk, U):
        dZ = self.h(Xk, U) * self.dt + self.sigma_W*np.random.normal(0, np.sqrt(self.dt), size=self.sigma_W.shape[0])
        dX = self.f(Xk, U) * self.dt + self.sigma_V*np.random.normal(0, np.sqrt(self.dt), size=self.sigma_V.shape[0])
        return dZ/self.dt, dX

    def run(self, X0, U):
        X = np.zeros((self.sigma_V.shape[0], U.shape[1]))
        Y = np.zeros((self.sigma_W.shape[0], U.shape[1]))
        Xk = np.array(X0)
        for k in range(U.shape[1]):
            X[:,k] = Xk
            Y[:,k], dX = self.update(Xk, U[:,k])
            Xk += dX
        return Y, X

class BodeDiagram(object):
    
    def __init__(self, bodes, Hz=False, dB=False, deg=False, filename='bode_diagram'):
        self.bodes = bodes
        for bode in self.bodes:
            if not (bode.Hz == Hz):
                bode.frequency *= (2*np.pi) if bode.Hz else 1/(2*np.pi)
            if not (bode.deg == deg):
                bode.phase *= (2*np.pi)/360 if bode.deg else 360/(2*np.pi)
            if not (bode.dB == dB):
                bode.magnitude = np.power(10,bode.magnitude/20) if bode.dB else 20*np.log10(bode.magnitude)
        self.frequency_label = 'Frequency ' + ( '[Hz]' if Hz else '[rad/s]')
        self.magnitude_label = 'Magnitude' + ( ' [dB]' if dB else '')
        self.phase_label = 'Phase ' + ( '[deg]' if deg else '[rad]')
        self.filename = filename if '.html' in filename else filename+'.html'

    def plot(self, fontsize=15):
        data = [None] * (2*len(self.bodes))
        for i, bode in enumerate(self.bodes):
            style = dict(
                name = bode.name,
                legendgroup = bode.name,
                line = dict(color = default_colors[np.mod(i,len(default_colors))])
            )

            magnitude_trace = go.Scatter(
                x = bode.frequency,
                y = bode.magnitude,
                yaxis = 'y2',
                **style
            )

            phase_trace = go.Scatter(
                x = bode.frequency,
                y = bode.phase,
                yaxis = 'y1',
                showlegend=False,
                **style
            )

            data[2*i] = magnitude_trace
            data[2*i+1] = phase_trace

        layout = go.Layout(
            title = 'Bode Diagram',
            font = dict(size=fontsize),
            xaxis = dict(
                title = self.frequency_label,
                type = 'log'
            ),
            yaxis1=dict(
                title = self.phase_label,
                # range = [-np.pi-0.1, 0.1],
                # tick0 = -np.pi,
                # dtick = np.pi/4,
                domain = [0,0.5],
                anchor = 'y1'
            ),
            yaxis2 = dict(
                scaleanchor = "x",
                title = self.magnitude_label,
                domain = [0.5,1],
                anchor = 'y2'
            )
        )

        fig = go.Figure(data=data, layout=layout)
        pyo.plot(fig, filename=self.filename)
        return fig
    
    @staticmethod
    def load_bode_file(filename, name=''):
        filename = filename if '.txt' in filename else filename+'.txt'
        name = filename.replace('.txt','') if name=='' else name
        frequency, magnitude, phase  = np.loadtxt(filename, unpack=True)
        return Struct(name=name, frequency=frequency, magnitude=magnitude, phase=phase, Hz=False, dB=False, deg=False)

    @staticmethod
    def create_bode_data(name, frequency, magnitude, phase, Hz=False, dB=False, deg=False):
        return Struct(name=name, frequency=frequency, magnitude=magnitude, phase=phase, Hz=Hz, dB=dB, deg=deg)

class StochasticSystem(object):
    def __init__(self, number_of_particles, model):
        self.d = model.d
        self.m = model.m
        self.sigma_V = model.sigma_V
        self.sigma_W = model.sigma_W
        self.f = model.f
        self.h = model.h
        self.particles = Particles(number_of_particles=number_of_particles, \
                                    X0_range=model.X0_range, states_constraints=model.states_constraints, \
                                    m=model.m)
        self.dt = model.dt
    
    def update(self, U, t):
        dX = self.f(self.particles.X, U, t)*self.dt + self.sigma_V*np.random.normal(0, np.sqrt(self.dt), size=self.particles.X.shape)
        self.particles.update(dX)
        self.particles.h = np.reshape(self.h(self.particles.X), [self.particles.Np, self.m])
        return
    
    def run(self, U, show_time=False):
        X = np.zeros([self.particles.Np, self.d, U.shape[1]])
        Y = np.zeros([self.particles.Np, self.m, U.shape[1]])
        for k in range(U.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            X[:,:,k] = self.particles.X
            self.update(U[:,k], k*self.dt)
            Y[:,:,k] = self.particles.h
            
        return Struct(X=X, Y=Y)

