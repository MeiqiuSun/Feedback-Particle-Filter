"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function
from abc import ABC, abstractmethod

import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

class Signal_Type(ABC):
    """Signal_Type:
        Notation Notes:
            d: number of states
            m: number of observations
        Initialize = Signal(self, dt, X0, sigma_V, sigma_W):
            dt: float, time step
            X0: 1D list of float with length d, initial states
            sigma_V: 1D list of float with the length d, standard deviation of process noise
            sigma_W: 1D list of float with the length m, standard deviation of observation noise
        Members:
            dt: float, size of time step in sec
            X0: numpy array with the shape of (d,), initial states
            sigma_V: numpy array with the shape of (d,), standard deviation of process noise
            sigma_W: numpy array with the shape of (m,), standard deviation of observation noise
        Methods:
            @abs f(self, X, t): return a numpy array with the shape of (d,), state tansition function maps states to rate change of states
            @abs h(self, X): return a numpy array with the shape of (m,), observation function maps states to observations
    """

    def __init__(self, dt, X0, sigma_V, sigma_W):
        self.dt = dt
        self.X0 = np.array(X0)
        self.sigma_V = np.array(sigma_V)
        self.sigma_W = np.array(sigma_W)
        return
    
    @abstractmethod
    def f(self, X, t):
        # dXdt = np.zeros(X.shape) 
        raise NotImplementedError

    @abstractmethod
    def h(self, X):
        # Y = np.zeros(self.sigma_W.shape)
        raise NotImplementedError

class Signal(object):
    """Signal:
        Notation Notes:
            m: number of observations(Y)
        Initialize = Signal(self, t, *args):
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            args*: tuple of numpy arrays, channels of signals
        Members:
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            value: numpy array with the shape of (m,T/dt+1), observations in time series
        Methods:
            plot(signal, fontsize=15, show=False): return an object of go.Figure, plot signal.Y in one frame
            create(cls, signal_type, T): return an object of Signal, create a signal based on signal_type with time length T
            __add__(self, other): return an object of Signal, combine two signal object
    """

    def __init__(self, t, *args):

        self.t = t
        self.dt = t[1]-t[0]
        self.T = t[-1]

        m = 0
        args = list(args)
        for i, arg in enumerate(args):
            if arg.ndim==1:
                args[i] = np.reshape(arg, (1,-1))
            m += args[i].shape[0]
        self.value = np.zeros((m, args[0].shape[1]))
        temp_m = 0
        for arg in args:
            self.value[temp_m:temp_m+arg.shape[0],:] = arg
            temp_m += arg.shape[0]
        return
    
    @staticmethod
    def plot(signal, fontsize=15, show=False):
        data = [None] * signal.value.shape[0]
        for m in range(signal.value.shape[0]):
            trace = go.Scatter(
                x = signal.t,
                y = signal.value[m,:],
                name = '$Y_{}$'.format(m+1)
            )
            data[m] = trace
        layout = go.Layout(
            title = 'signal',
            font = dict(size=fontsize),
            xaxis = dict(title='time [s]')
        )
        fig = go.Figure(data=data, layout=layout)
        if show:
            pyo.plot(fig, filename='signal_test.html', include_mathjax='cdn')
        return fig

    @staticmethod
    def quantile_plot(signals, names, continuous=False, fontsize=15, show=False):
        data = [None] * len(signals)
        if not names:
            names = [r'$Y_{}$'.format(i) for i in range(len(signals))]

        for j in range(len(signals)):
            signal = signals[j]
            mean = np.mean(signal.value, axis=0)
            # mean = np.quantile(signal.value, 0.5, axis=0)
            quantile3 = np.quantile(signal.value, 0.75, axis=0)
            quantile1 = np.quantile(signal.value, 0.25, axis=0)
            trace = go.Scatter(
                x = signal.t,
                y = mean,
                name = names[j],
                error_y = dict(
                    type='data',
                    array=quantile3-mean,
                    arrayminus=mean-quantile1,
                    visible=True
                )
            )
            data[j] = trace
        layout = go.Layout(
            title = 'signal',
            font = dict(size=fontsize),
            xaxis = dict(title='time [s]')
        )
        fig = go.Figure(data=data, layout=layout)
        if show:
            pyo.plot(fig, filename='signal_test.html', include_mathjax='cdn')
        return fig
    @staticmethod
    def stack(signal1, signal2, *args):
        signal = signal1 + signal2
        if len(args) != 0:
            for arg in args:
                signal += arg
        return signal

    @classmethod
    def create(cls, signal_type, T):
        # Create time signal
        T = signal_type.dt*int(T/signal_type.dt)
        t = np.arange(0, T+signal_type.dt, signal_type.dt)

        dX = np.zeros([signal_type.X0.shape[0], t.shape[0]])
        X = np.zeros(dX.shape)
        dZ = np.zeros([signal_type.sigma_W.shape[0], t.shape[0]])
        Y = np.zeros(dZ.shape)
        for k in range(t.shape[0]):
            # Create state signals: dX = f(X, t)*dt + sigma_B*dB
            if k==0:
                dX[:,k] = signal_type.X0
                X[:,k] = signal_type.X0
            else:
                dX[:,k] = signal_type.f(X[:,k-1], t[k-1])*signal_type.dt + signal_type.sigma_V*np.random.normal(0, np.sqrt(signal_type.dt), size=signal_type.sigma_V.shape[0])
                X[:,k] = X[:,k-1] + dX[:,k]
            # Create observation increment signals: dZ = h(X)*dt + sigma_W*dW
            dZ[:,k] = signal_type.h(X[:,k])*signal_type.dt + signal_type.sigma_W*np.random.normal(0, np.sqrt(signal_type.dt))
            # Create observation signals: Y = dZ/dt = h(X) + white noise with std sigma_W
            Y[:,k] = dZ[:,k] / signal_type.dt

        return Signal(t, Y), Signal(t, X)

    def __add__(self, other):
        assert self.dt == other.dt, "time scales (dt) are different"
        assert self.T == other.T, "time lengths (T) are different"        
        return Signal(self.t, self.value, other.value)

    def __iadd__(self, other):      
        self = self + other
        return self

class Linear(Signal_Type):
    def __init__(self, dt):        
        Signal_Type.__init__(self, dt=dt, X0=[1,0], sigma_V=[0.1, 0.1], sigma_W=[0.001, 0.001])
        pass

    def f(self, X, t):
        A = np.array([[ 0,  1],\
                      [-1, -1]])
        return np.matmul(A,X)

    def h(self, X):
        H = np.array([[1,0],\
                      [0,1]])
        return np.matmul(H,X)

class Sinusoids(Signal_Type):
    # SNR: Signal to Noise ratio
    def __init__(self, dt=0.01, amp=[[1]], freq=[1], theta0=[[0]], sigma_V=[0.1], SNR=[100]):
        assert len(amp)==len(theta0)==len(SNR), \
            "rows of amp ({}) and theta0 ({}) and length of SNR ({}) should be number of output m ({})".format(len(amp),len(theta0),len(SNR),len(SNR))
        assert len(amp[0])==len(theta0[0])==len(freq)==len(sigma_V),\
            "columns of amp ({}) and theta0 ({}) and length of sigma_V ({}) should be number of frequencies ({})".format(len(amp[0]),len(theta0[0]),len(sigma_V),len(freq))
        
        self.amp = np.array(amp)        # amplitude
        self.freq = np.array(freq)      # frequency
        self.theta0 = np.array(theta0)  # theta0
        sigma_W = np.zeros(len(SNR))    # observation noise
        for m in range(len(SNR)):
            sigma_W[m] = np.max(self.amp[m,:])/np.power(10, SNR[m]/10)
        
        Signal_Type.__init__(self, dt=dt, X0=np.zeros(self.freq.shape), sigma_V=np.array(sigma_V), sigma_W=sigma_W)
        pass
    
    def f(self, X, t):
        X_dot = 2*np.pi*self.freq
        return X_dot
    
    def h(self, X):
        Y = np.zeros(self.sigma_W.shape[0])
        for m in range(self.sigma_W.shape[0]):
            for k in range(self.freq.shape[0]):
                Y[m] += self.amp[m,k]*np.sin(X[k]+self.theta0[m,k])
        return Y

if __name__ == "__main__":
    
    T = 10
    sampling_rate = 100 # Hz
    dt = 1./sampling_rate
    # signal_type = Sinusoids(dt, amp=[[1]], freq=[1], theta0=[[0]], sigma_V=[0.01], SNR=[100])
    signal_type = Sinusoids(dt, amp=[[2,1],[1,3]], freq=[1,10], theta0=[[0, np.pi/4],[np.pi/2, np.pi]], sigma_V=[0.01, 0.1], SNR=[100,100])
    signal, _ = Signal.create(signal_type=signal_type, T=T)
    fig = Signal.plot(signal)
    pyo.plot(fig, filename='signal_test.html', include_mathjax='cdn')
    pass