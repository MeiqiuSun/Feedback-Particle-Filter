"""
Created on Tue Jul. 09, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import sympy
import numpy as np
import math
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.io as pio

from FPF import FPF, Model, Galerkin, Figurely
from Tool import Struct

ppg_raw = [1077,1095.27676998271,1150.39696200268,1253.64011384634,1410.01430649103,1614.23738042021,1852.88862847947,2107.19963850395,2354.10827203910,2571.83504395768,2747.58603743220,2878.78047565119,2967.12445239160,3018.75510162723,3042.94721680823,3047.45001341082,3037.62026345592,3018.14128598677,2991.80738548651,2959.68011205818,2924.12507822977,2885.14270668177,2843.86782834153,2802.08903767062,2759.59973773619,2716.75011175682,2673.89981522322,2630.12661808970,2584.21488466353,2533.75439619717,2480.71246051141,2423.10076799285,2360.92007271860,2294.87110683952,2226.11520653275,2155.41275815825,2085.84032425344,2019.82560572195,1959.08788519998,1905.61966740180,1860.15018328118,1823.59858526554,1799.16523439130,1785.68124422590,1781.40496155451,1785.65000894081,1795.60758813888,1810.23123770638,1825.88000834451,1843.33369762174,1863.68739383102,1877.70370775466,1882.41001847768,1880.67124988824,1874.06744888836,1862.89381761287,1847.14991953269,1829.89768909817,1810.34993145378,1788.72101296379,1765.94996721702,1742.03613261809,1717.39751028193,1692.75877484727,1668.68005245276,1644.48098438338,1619.84239547012,1595.97728854980,1572.41003606020,1548.84233116767,1525.27517210550,1501.70741230815,1478.27021279132,1455.77367020325,1433.54980777827,1412.12496274662,1390.69993145582,1369.27500149014,1348.40755237670,1328.05378822197,1308.39978542052,1289.88870123678,1272.51984025750,1256.29379973178,1241.20988734577,1229.73332273703,1217.97235166145,1206.26252309710,1195.54992698555,1185.27003397509,1176.98743219884,1169.12989658769,1160.98994575908,1153.99245954403,1146.99245931931,1138.20624497094,1127.13495738213,1112.99225070036,1097.13740724184,1083.85480956071,1080]
ppg_raw = np.array(ppg_raw)
ppg_raw = (ppg_raw-np.min(ppg_raw))/(np.max(ppg_raw)-np.min(ppg_raw))

class PPG(object):
    def __init__(self, t, freq, amp=1):
        self.t = t
        ppg = ppg_raw*amp
        reps = int(np.ceil(self.t[-1]*freq))
        ppg = np.tile(ppg, reps=reps)
        t = np.linspace(0, reps/freq, ppg.shape[0])
        self.Y = np.reshape(np.interp(self.t, t, ppg), (1,-1))

class Model_PPG(Model):
    def __init__(self, an, bn, amp_range, freq_range, sigma_V, sigma_W):
        self.an = an
        self.bn = bn
        self.amp_min = amp_range[0]
        self.amp_max = amp_range[1]
        self.freq_min = freq_range[0]
        self.freq_max = freq_range[1]
        Model.__init__(self, m=1, d=2, X0_range=[[0,2*np.pi],amp_range], 
                        states=['theta','r'], states_label=[r'$\theta$',r'$r$'],
                        sigma_V=sigma_V, sigma_W=sigma_W)
        self.ppg_function = np.poly1d(np.polyfit(np.linspace(0, 2*np.pi, len(ppg_raw)), ppg_raw, self.order))
    
    def states_constraints(self, X):
        X[:,0] = np.mod(X[:,0], 2*np.pi)
        X[X[:,1]<self.amp_min,1] = self.amp_min*np.ones(X[X[:,1]<self.amp_min,1].shape)
        X[X[:,1]>self.amp_max,1] = self.amp_max*np.ones(X[X[:,1]>self.amp_max,1].shape)
        return X
    
    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        Y = X[:,1]*self.ppg_function(X[:,0])
        return Y

class Galerkin_PPG(Galerkin):
    def __init__(self, states, order):
        Galerkin.__init__(self, states, m=1, L=order+1)
        # self.states = ['self.'+state for state in states]
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        for l in range(self.L):
            self.psi_list[l] = 'self.r*self.theta**{}/{}'.format(l,math.factorial(l))
        self.psi_string += (', '.join(self.psi_list)).replace('sympy','np')
        self.psi_string += ']'
        
        # initialize symbols
        exec(', '.join(self.states)+' = sympy.symbols("'+', '.join(self.states)+'")')
        
        # calculate grad_psi and grad_grad_psi
        self.grad_psi_string = '['
        self.grad_grad_psi_string = '['
        for l in range(self.L):
            self.grad_psi_string += '['
            self.grad_grad_psi_string += '['
            for d1 in range(self.d):
                exec('self.result = sympy.diff('+self.psi_list[l]+','+self.states[d1]+')')
                self.grad_psi_list[l][d1] = self.sympy_executable_result()
                
                self.grad_grad_psi_string += '['
                for d2 in range(self.d):
                    exec('self.result = sympy.diff('+self.grad_psi_list[l][d1]+','+self.states[d2]+')')
                    self.grad_grad_psi_list[l][d1][d2] = self.sympy_executable_result()
                self.grad_grad_psi_string += ((', ').join(self.grad_grad_psi_list[l][d1])).replace('sympy','np')
                self.grad_grad_psi_string += '], '
            
            self.grad_psi_string += ((', ').join(self.grad_psi_list[l])).replace('sympy','np')
            self.grad_psi_string += '], '
            self.grad_grad_psi_string = self.grad_grad_psi_string[:-2] + '], '

        self.grad_psi_string = self.grad_psi_string[:-2]+']'
        self.grad_grad_psi_string = self.grad_grad_psi_string[:-2]+']' 
    
    def sympy_executable_result(self):
        self.result = str(self.result)
        if 'cos' in self.result:
            self.result = self.result.replace('cos','sympy.cos')
        if 'sin' in self.result:
            self.result = self.result.replace('sin','sympy.sin')
        return self.result

    def psi(self, X):
        self.split(X)
        exec("self.result = "+self.psi_string)
        trial_functions = self.result
        return np.array(trial_functions)
        
    def grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_psi_string)
        grad_trial_functions = np.zeros([self.L, self.d, X.shape[0]])
        for l in range(self.L):
            for d in range(self.d):
                grad_trial_functions[l,d,:] = self.result[l][d]
        return grad_trial_functions

    def grad_grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_grad_psi_string)
        grad_grad_trial_functions = np.zeros([self.L, self.d, self.d, X.shape[0]])
        for l in range(self.L):
            for d1 in range(self.d):
                for d2 in range(self.d):
                    grad_grad_trial_functions[l,d1,d2,:] = self.result[l][d1][d2]
        return grad_grad_trial_functions

if __name__ == '__main__':
    dt = 1/500
    T = 10
    t = np.arange(0, T+dt, dt)
    freq = 2
    amp = 3
    ppg = PPG(t=t, freq=freq, amp=amp)

    Np = 1000
    model = Model_PPG(amp_range=[amp*0.1, amp*1.1], freq_range=[freq*0.9, freq*1.1], sigma_V=[0,0.1], sigma_W=[0.1])
    galerkin = Galerkin_PPG(model.states, order=model.order)
    feedback_particle_filter = FPF(number_of_particles=Np, model=model, galerkin=galerkin, dt=dt)
    filtered_signal = feedback_particle_filter.run(ppg.Y, show_time=True)
    
    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_histogram=False, plot_c=False)
    figure = Figurely(fig_property=fig_property, signal=ppg, filtered_signal=filtered_signal)
    figure.plot_figures(show=True)







    # trace = go.Scatter(
    #     x = ppg.t,
    #     y = ppg.Y,
    #     name = 'ppg'
    # )
    # data = [trace]
    # layout = go.Layout(
    #     title = "PPG Pulses with {} Hz".format(freq),
    #     font=dict(family='Courier New, monospace', size=30),
    #     xaxis = dict(title='time [sec]', titlefont=dict(size=25), tickfont=dict(size=20)),
    #     yaxis = dict(tickfont=dict(size=20))
    # )

    # fig = go.Figure(data=data, layout=layout)
    # pyo.plot(fig, filename='temp_plot.html', auto_open=True, include_mathjax='cdn')
    pass