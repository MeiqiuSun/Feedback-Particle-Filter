"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import numpy as np
import bisect

from plotly import tools
import plotly.offline as pyo
import plotly.graph_objs as go

class Struct(dict):
    """create a struct"""
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__.update(kwargs)

def isiterable(object):
    """check if object is iterable"""
    try:
        iter(object)
    except TypeError: 
        return False
    return True

def find_limits(signals, scale='normal'):
    """find the maximum and minimum of plot limits"""
    maximums = np.zeros(signals.shape[0])
    minimums = np.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        maximums[i] = np.max(signals[i])
        minimums[i] = np.min(signals[i])

    if scale=='log':
        signal_range = np.log10(np.max(maximums))-np.log10(np.min(minimums))
        minimum = np.power(10, np.log10(np.min(minimums))-signal_range/100)
        maximum = np.power(10, np.log10(np.max(maximums))+signal_range/100)
    else:
        signal_range = np.max(maximums)-np.min(minimums)
        minimum = np.min(minimums)-signal_range/10
        maximum = np.max(maximums)+signal_range/10
    return minimum, maximum

def find_closest(num, seq, pos=False):
    index = bisect.bisect_left(a=seq, x=num)
    if index == len(seq):
        raise ValueError
    if pos:
        return index
    else :
        return seq[index]

class BodeDiagram(object):

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
                line = dict(color = self.default_colors[np.mod(i,len(self.default_colors))])
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

if __name__ == "__main__":
    # bode1 = BodeDiagram.load_bode_file('bode_f_band_noiseless.txt')
    # bode2 = BodeDiagram.load_bode_file('bode_f_fix_noiseless.txt')
    # bodeDiagram = BodeDiagram([bode1, bode2], dB=True, Hz=False)
    # bodeDiagram.plot()
    # print(bodeDiagram.frequency_label)
    # amp_ranges=[[0,1],[0,2],[0,3],[0,4]]
    # sigma_B=[[0, 1, 2, 3, 4]]
    # print(amp_ranges)
    # print(sigma_B)
    # sigma_B = np.array(sigma_B)
    # print(sigma_B, type(sigma_B))
    # sigma_B = np.array(sigma_B)
    # print(sigma_B, type(sigma_B))
    # known_observations = [2,3]
    # amp_ranges = [amp_ranges[m-1] for m in known_observations]
    # known_observations.insert(0,0)
    # sigma_B = [sigma_B[m] for m in known_observations]
    # print(amp_ranges)
    # print(sigma_B)

    # array = np.loadtxt('bode1.txt')
    # array[array<-np.pi] += 2*np.pi 
    # np.savetxt('bode1.txt', array)
    # fig_property = Struct(plot_signal=True, plot_X=False)

    # if ('plot_signal' in fig_property.__dict__) and fig_property.plot_signal:
    #     print('plot_signal')
        
    # if ('plot_X' in fig_property.__dict__) and fig_property.plot_X:
    #     print('plot_X')
        
    # if ('plot_histogram' in fig_property.__dict__) and fig_property.plot_histogram:
    #     print('plot_histogram')
    # trace1 = go.Scatter(
    #     x=[0, 1, 2],
    #     y=[10, 11, 12]
    # )
    # trace2 = go.Scatter(
    #     x=[2, 3, 4],
    #     y=[100, 110, 120],
    # )
    # trace3 = go.Scatter(
    #     x=[3, 4, 5],
    #     y=[100, 1100, 1200],
    # )
    # fig = tools.make_subplots(rows=3, cols=1, specs=[[{}], [{}], [{}]],
    #                         shared_xaxes=True, shared_yaxes=True,
    #                         vertical_spacing=0.1)
    # fig.append_trace(trace1, 3, 1)
    # fig.append_trace(trace2, 2, 1)
    # fig.append_trace(trace3, 1, 1)

    # fig['layout'].update(height=600, width=600, title='Stacked Subplots with Shared X-Axes')
    # pyo.plot(fig, filename='stacked-subplots-shared-xaxes')
    if []:
        print(True)
    else:
        print(False)
    pass
    