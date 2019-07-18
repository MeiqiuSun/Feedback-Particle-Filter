"""
Created on Wed Feb. 13, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import numpy as np

from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go

from Tool import Struct, isiterable, find_limits, find_closest
from FPF import FPF, Model, Galerkin

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter

class CFPF(object):
    """CFPF: Coupled Feedback Partilce Filter
        Initialize = CFPF(fpfs, sigma_W, dt)
            fpfs: 1D list of Struct, properties of FPF
            sigma_W: 1D list of float with legnth m, standard deviations of noise of observations(Y)
            dt: float, size of time step in sec
        Members = 
            Nc: number of channels
            fpfs: 1D list of FPF with length Nc, a list of FPF for each channel
            dt: float, size of time step in sec
            h_hat: numpy array with the shape of (m,), filtered observations
        Methods =
            calculate_h_hat(): Summation of h_hat for all FPF channels
            update(Y): Update each FPF channel with new observation data Y
            run(Y): Run CFPF with time series observations(Y)
    """

    def __init__(self, fpfs, sigma_W, dt):
        self.Nc = len(fpfs)
        self.fpfs = [None]*self.Nc
        for j in range(self.Nc):
            self.fpfs[j] = FPF(number_of_particles=fpfs[j].number_of_particles,
                                model=fpfs[j].model, galerkin=fpfs[j].galerkin, dt=dt)
        self.h_hat = np.zeros(len(sigma_W))
        self.dt = dt
        return

    def calculate_h_hat(self):
        """Summation of h_hat for all FPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for j in range(self.Nc):        
            h_hat += self.fpfs[j].h_hat
        return h_hat
    
    def update(self, Y):
        """Update each FPF channel with new observation data y"""
        dU = [None] * len(self.fpfs)
        for j in range(self.Nc):
            dU[j] = self.fpfs[j].update(Y-self.h_hat+self.fpfs[j].h_hat)
        self.h_hat = self.calculate_h_hat()
        return dU
    
    def run(self, Y, show_time=False):
        """Run CFPF with time series observations(y)"""
        h_hat = np.zeros(Y.shape)
        fpfs = [None] * self.Nc
        
        for j in range(self.Nc):
            fpfs[j] = Struct(states_label=self.fpfs[j].states_label,
                                h_hat=np.zeros(Y.shape), 
                                h=np.zeros((self.fpfs[j].particles.Np, self.fpfs[j].m, Y.shape[1])),
                                X=np.zeros((self.fpfs[j].particles.Np, self.fpfs[j].d, Y.shape[1])))
            
        for k in range(Y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            
            h_hat[:,k] = self.h_hat
            for j in range(self.Nc):
                fpfs[j].h_hat[:,k] = self.fpfs[j].h_hat
                fpfs[j].h[:,:,k] = self.fpfs[j].particles.h
                fpfs[j].X[:,:,k] = self.fpfs[j].particles.X
            self.update(Y[:,k])

        return Struct(h_hat=h_hat, fpfs=fpfs)

class Figure(object):

    default_colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
        ]

    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
        self.figs = dict()

        # set properties for plot_X and histogram if needed
        if ('plot_X' in self.fig_property.__dict__) and self.fig_property.plot_X:
            # set sampled index
            Np = [None] * len(filtered_signal.fpfs)
            self.sampled_index = [None] * len(filtered_signal.fpfs)
            for j in range(len(filtered_signal.fpfs)):
                Np[j] = filtered_signal.fpfs[j].X.shape[0]
                if 'particles_ratio' in self.fig_property.__dict__ :
                    self.sampled_index[j] = np.random.choice(Np[j], size=int(Np[j]*self.fig_property.particles_ratio), replace=False)
                else :
                    self.sampled_index[j] = np.array(range(Np[j])) if Np[j]<10 else np.random.choice(Np[j], size=10, replace=False)

            # set histogram property
            if not 'histogram' in self.fig_property.__dict__:
                self.fig_property.histogram = Struct(t=['-1'], n_bins=100)
                # self.fig_property.histogram = Struct(t=[2.5,0,10,'-1'], n_bins=100)
            else:
                if not 'n_bins' in self.fig_property.histogram.__dict__:
                    self.fig_property.histogram = Struct(t=self.fig_property.histogram.t, n_bins=100)
                if not 't' in self.fig_property.histogram.__dict__:
                    self.fig_property.histogram = Struct(t=['-1'], n_bins=self.fig_property.histogram.n_bins)

            if len(self.fig_property.histogram.t) == 0:
                self.fig_property.histogram = Struct(length=0, n_bins=self.fig_property.histogram.n_bins)
            else:
                t_index = [None] * len(self.fig_property.histogram.t)
                for k, t in enumerate(self.fig_property.histogram.t):
                    t_index[k] = np.mod(signal.t.shape[0]+int(t), signal.t.shape[0]) if isinstance(t, str) \
                        else find_closest(t, signal.t.tolist(), pos=True)
                t_index = list(dict.fromkeys(t_index))
                t_index.sort()
                self.fig_property.histogram = Struct(
                    t = signal.t[t_index], 
                    t_index = t_index, 
                    length = len(t_index), 
                    n_bins = self.fig_property.histogram.n_bins
                )

    def plot_figures(self, show=False):
        if ('plot_signal' in self.fig_property.__dict__) and self.fig_property.plot_signal:
            self.figs['signal'] = self.plot_signal()
        if ('plot_X' in self.fig_property.__dict__) and self.fig_property.plot_X:
            for j in range(len(self.filtered_signal.fpfs)):
                self.figs['X{}'.format(j+1)] = self.plot_X(j)
        if show:
            self.show()
    
    def plot_signal(self):
        print("plot_signal")
        specs = [ [{'rowspan':3}] * self.signal.Y.shape[0],
                [None] * self.signal.Y.shape[0],
                [None] * self.signal.Y.shape[0],
                [{}] * self.signal.Y.shape[0],
                [{}] * self.signal.Y.shape[0] ]
        fig = tools.make_subplots(
            rows = len(specs), 
            cols = self.signal.Y.shape[0],
            specs = specs,
            shared_xaxes = True
        )

        for m in range(self.signal.Y.shape[0]):
            fig.append_trace(
                trace=go.Scatter(
                    x = self.signal.t,
                    y = self.signal.Y[m,:],
                    name = 'signal',
                    line = dict(color='grey')
                ), row=1, col=m+1)
            fig.append_trace(
                trace=go.Scatter(
                    x = self.signal.t,
                    y = self.filtered_signal.h_hat[m,:],
                    name = 'estimation',
                    line = dict(color='magenta')
                ), row=1, col=m+1)
            fig.append_trace(
                trace=go.Scatter(
                    x = self.signal.t,
                    y = self.filtered_signal.h_hat[m,:] - self.signal.Y[m,:],
                    name = 'error',
                    line = dict(color='magenta')
                ), row=len(specs)-1, col=m+1)
            for j in range(len(self.filtered_signal.fpfs)):
                fig.append_trace(
                    trace=go.Scatter(
                        x = self.signal.t,
                        y = self.filtered_signal.fpfs[j].h_hat[m,:],
                        name = r'$\hat h_{}$'.format(j+1),
                        line = dict(color=self.default_colors[np.mod(j,len(self.default_colors))])
                    ), row=len(specs), col=m+1)
        error_yaxis = {'yaxis{}'.format(self.signal.Y.shape[0]+1):dict(title='error')}
        fig['layout'].update(
            showlegend = False,
            font = dict(size=self.fig_property.fontsize),
            xaxis = dict(title='time [s]'),
            **error_yaxis
        )
        return fig

    def plot_X(self, j):
        print("plot_X^{}".format(j+1))
        if self.fig_property.histogram.length==0 :
            row_spec = [{}]
        else:
            row_spec = [{'colspan':self.fig_property.histogram.length}]
            row_spec.extend([None] * (self.fig_property.histogram.length-1))
            row_spec.extend([{}] * self.fig_property.histogram.length)
        specs = [row_spec] * self.filtered_signal.fpfs[j].X.shape[1]
        
        fig = tools.make_subplots(
            rows = self.filtered_signal.fpfs[j].X.shape[1],
            cols = max(2*self.fig_property.histogram.length,1),
            specs = specs,
            shared_xaxes = True,
            shared_yaxes = True
        )

        limits = np.zeros((self.filtered_signal.fpfs[j].X.shape[1],2))
        for d in range(self.filtered_signal.fpfs[j].X.shape[1]):
            limits[d,:] = find_limits(self.filtered_signal.fpfs[j].X[:,d,:])
            for color_index, p in enumerate(self.sampled_index[j]):
                fig.append_trace(
                    trace=go.Scatter(
                        x = self.signal.t,
                        y = self.filtered_signal.fpfs[j].X[p,d,:],
                        mode = 'markers',
                        name = '$X^{}$'.format(p),
                        line = dict(color=self.default_colors[np.mod(color_index,len(self.default_colors))]),
                        legendgroup = 'X^{}'.format(p),
                        showlegend = True if d==0 else False
                    ), row=d+1, col=1)
        
            for k in range(self.fig_property.histogram.length):
                trace = go.Histogram(
                    y = self.filtered_signal.fpfs[j].X[:,d,self.fig_property.histogram.t_index[k]],
                    histnorm = 'probability',
                    nbinsy = self.fig_property.histogram.n_bins,
                    marker=dict(color='rgb(50, 50, 125)'),
                    name = 't={}'.format(self.fig_property.histogram.t[k]),
                    showlegend = False
                )
                fig.append_trace(trace = trace, row=d+1, col=self.fig_property.histogram.length+k+1)

        xaxis = dict()
        t_lines = []
        for k in range(self.fig_property.histogram.length):
            xaxis['xaxis{}'.format(self.fig_property.histogram.length+k+1)] = dict(title='t={}'.format(self.fig_property.histogram.t[k]), range=[0,1], dtick=0.25)
            for d in range(self.filtered_signal.fpfs[j].X.shape[1]):
                t_lines.append(dict(
                    type = 'line', 
                    xref = 'x1',
                    yref = 'y{}'.format(d+1),
                    x0 = self.fig_property.histogram.t[k],
                    x1 = self.fig_property.histogram.t[k],
                    y0 = limits[d,0],
                    y1 = limits[d,1],
                    line = dict(color='rgb(50, 50, 125)', width=3)
                ))
        yaxis = dict()
        for d in range(self.filtered_signal.fpfs[j].X.shape[1]):
            yaxis['yaxis{}'.format(d+1)] = dict(title=self.filtered_signal.fpfs[j].states_label[d], range=limits[d,:])
        fig['layout'].update(
            font = dict(size=self.fig_property.fontsize),
            showlegend = False,
            shapes = t_lines,
            xaxis1 = dict(title='time [s]'),
            **xaxis,
            **yaxis
        )
        return fig
    
    def show(self):
        for name, fig in self.figs.items():
            pyo.plot(fig, filename=name+'.html', include_mathjax='cdn')

if __name__ == "__main__":
    T = 5.5
    sampling_rate = 100 # Hz
    dt = 1./sampling_rate

    pass