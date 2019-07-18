"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import numpy as np

import plotly.offline as pyo
import plotly.graph_objs as go

class Signal(object):
    """Signal:
        Notation Notes:
            d: dimension of states(X)
            m: number of observations(Y)
        Initialize = Signal(f, h, sigma_B, sigma_W, X0, dt, T)
            f(X, t): state tansition function maps states(X) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            sigma_V: 1D list with legnth d, standard deviation of noise of states(X)
            sigma_W: 1D list with legnth m, standard deviation of noise of observations(Y)
            X0: 1D list with length d, initial condition of states(X)
            dt: float, size of time step in sec
            T: float or int, end time in sec
        Members =
            dt: float, size of time step in sec
            T: float, end time in sec, which is also an integer mutiple of dt
            t: numpy array with the shape of (T/dt+1,), i.e. 0, dt, 2dt, ... , T-dt, T
            f(X, t): state tansition function maps states(X) to states(X_dot)
            h(X): observation function maps states(X) to observations(Y)
            sigma_V: numpy array with the shape of (d,1), standard deviation of noise of states(X)
            sigma_W: numpy array with the shape of (m,1), standard deviation of noise of observations(Y)
            X: numpy array with the shape of (d,T/dt+1), states in time series
            dX: numpy array with the shape of (d,T/dt+1), increment of states in time series
            Y: numpy array with the shape of (m,T/dt+1), observations in time series
            dZ: numpy array with the shape of (m,T/dt+1), increment of integral of observations in time series
    """
    def __init__(self, signal_type=None, T=None):
        if signal_type and T:
            self.dt = signal_type.dt
            self.T = self.dt*int(T/self.dt)
            self.t = np.arange(0, self.T+self.dt, self.dt)

            self.f = signal_type.f
            self.h = signal_type.h
            
            self.sigma_V = np.array(signal_type.sigma_V)
            self.sigma_W = np.array(signal_type.sigma_W)

            self.dX = np.zeros([signal_type.X0.shape[0], self.t.shape[0]])
            self.X = np.zeros(self.dX.shape)
            self.dZ = np.zeros([self.sigma_W.shape[0], self.t.shape[0]])
            self.Y = np.zeros(self.dZ.shape)
            for k in range(self.t.shape[0]):
                # dX = f(X, t)*dt + sigma_B*dB
                if k==0:
                    self.dX[:,k] = np.array(signal_type.X0)
                    self.X[:,k] = np.array(signal_type.X0)
                else:
                    self.dX[:,k] = self.f(self.X[:,k-1], self.t[k-1])*self.dt + self.sigma_V*np.random.normal(0, np.sqrt(self.dt), size=self.sigma_V.shape[0])
                    self.X[:,k] = self.X[:,k-1] + self.dX[:,k]
                # dZ = h(X)*dt + sigma_W*dW
                self.dZ[:,k] = self.h(self.X[:,k])*self.dt + self.sigma_W*np.random.normal(0, np.sqrt(self.dt))
                # Y = dZ/dt = h(X) + white noise with std sigma_W
                self.Y[:,k] = self.dZ[:,k] / self.dt
        else:
            self.dt = None
            self.T = None
            self.t = None
            self.sigma_V = None
            self.sigma_W = None
            self.dX = None
            self.X = None
            self.dZ = None
            self.Y = None
    
    @staticmethod
    def plot(signal, fontsize=15, show=False):
        data = [None] * signal.Y.shape[0]
        for m in range(signal.Y.shape[0]):
            trace = go.Scatter(
                x = signal.t,
                y = signal.Y[m,:],
                name = '$Y_{}$'.format(m+1)
            )
            data[m] = trace
        layout = go.Layout(
            title = 'signal',
            font = dict(size=fontsize),
            xaxis = dict(title='time')
        )
        fig = go.Figure(data=data, layout=layout)
        if show:
            pyo.plot(fig, filename='signal_test.html', include_mathjax='cdn')
        return fig

    @classmethod
    def create(cls, t, *args):
        signal = Signal()
        signal.t = t
        m = 0
        for arg in args:
            m += arg.shape[0]
        signal.Y = np.zeros((m, args[0].shape[1]))
        temp_m = 0
        for arg in args:
            signal.Y[temp_m:temp_m+arg.shape[0],:] = arg
            temp_m += arg.shape[0]
        return signal

    def __add__(self, other):
        assert self.dt==other.dt, "time scales (dt) are different"
        assert self.T==other.T, "time length (T) are different"
        signal = Signal()
        
        signal.t = self.t
        signal.dt = self.dt
        signal.T = self.T

        signal.sigma_V = np.zeros(self.sigma_V.shape[0]+other.sigma_V.shape[0])
        signal.sigma_W = np.zeros(self.sigma_W.shape[0]+other.sigma_W.shape[0])
        signal.dX = np.zeros([signal.sigma_V.shape[0], signal.t.shape[0]])
        signal.X = np.zeros(signal.dX.shape)
        signal.dZ = np.zeros([signal.sigma_W.shape[0], signal.t.shape[0]])
        signal.Y = np.zeros(signal.dZ.shape)

        for d in range(self.sigma_V.shape[0]):
            signal.sigma_V[d] = self.sigma_V[d]
            signal.dX[d,:] = self.dX[d,:]
            signal.X[d,:] = self.X[d,:]
        for d in range(other.sigma_V.shape[0]):
            signal.sigma_V[self.sigma_V.shape[0]+d] = other.sigma_V[d]
            signal.dX[self.sigma_V.shape[0]+d,:] = other.dX[d,:]
            signal.X[self.sigma_V.shape[0]+d,:] = other.X[d,:]
        
        for m in range(self.sigma_W.shape[0]):
            signal.sigma_W[m] = self.sigma_W[m]
            signal.dZ[m,:] = self.dZ[m,:]
            signal.Y[m,:] = self.Y[m,:]
        for m in range(other.sigma_W.shape[0]):
            signal.sigma_W[self.sigma_W.shape[0]+m] = other.sigma_W[m]
            signal.dZ[self.sigma_W.shape[0]+m,:] = other.dZ[m,:]
            signal.Y[self.sigma_W.shape[0]+m,:] = other.Y[m,:]
        
        return signal

class Linear(object):
    def __init__(self, dt):
        # N states of state noises
        self.sigma_V = [0.1, 0.1]
        # M states of observation noises
        self.sigma_W = [0.001, 0.001]
        # initial state condition
        self.X0 = [1, 0]
        # sampling time
        self.dt = dt

    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        A = np.array([[ 0,  1],\
                      [-1, -1]])
        return np.matmul(A,X)

    # h(X): observation function maps states(X) to observations(Y)
    def h(self, X):
        H = np.array([[1,0],\
                      [0,1]])
        return np.matmul(H,X)

class LTI(object):
    def __init__(self, A, B, C, D, sigma_V, sigma_W, dt):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.D = np.array(D)
        self.sigma_V = np.array(sigma_V)
        self.sigma_W = np.array(sigma_W)
        self.dt = dt

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

class Sinusoidals(object):
    # SNR: Signal to Noise ratio
    def __init__(self, dt=0.01, amp=[[1]], freq=[1], theta0=[[0]], sigma_V=[0.1], SNR=[100]):
        assert len(amp)==len(theta0)==len(SNR), \
            "rows of amp ({}) and theta0 ({}) and length of SNR ({}) should be number of output m ({})".format(len(amp),len(theta0),len(SNR),len(SNR))
        assert len(amp[0])==len(theta0[0])==len(freq)==len(sigma_V),\
            "columns of amp ({}) and theta0 ({}) and length of sigma_V ({}) should be number of frequencies ({})".format(len(amp[0]),len(theta0[0]),len(sigma_V),len(freq))
        
        # sampling time
        self.dt = dt

        # amplitude
        self.amp = np.array(amp)
        # frequency
        self.freq = np.array(freq)
        # theta0
        self.theta0 = np.array(theta0)

        # initial states condition 
        self.X0 = np.zeros(self.freq.shape)
        # for m in range(self.amp.shape[0]):
        #     for k in range(self.amp.shape[1]):
        #         # print(m,k,m*self.amp.shape[1]+k)
        #         self.X0[m*self.amp.shape[1]+k] = np.mod(theta0[m][k], 2*np.pi)
        
        # k states of frequency noises
        self.sigma_V = np.array(sigma_V)
        
        # m states of observation noises
        self.sigma_W = np.zeros(len(SNR))
        for m in range(len(SNR)):
            self.sigma_W[m] = np.max(self.amp[m,:])/np.power(10, SNR[m]/10)
    
    # f(X, t): state tansition function maps states(X) to states(X_dot)
    def f(self, X, t):
        X_dot = 2*np.pi*self.freq
        # for k in range(self.freq.shape[0]):
        #     X_dot[k::self.freq.shape[0]] = 2*np.pi*self.freq[k]
        return X_dot
    
    # h(X): observation function maps states(X) to observations(Y)
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

    signal_type = Sinusoidals(dt, amp=[[2,1],[1,3]], freq=[1,10], theta0=[[0, np.pi/4],[np.pi/2, np.pi]], sigma_V=[0.01, 0.1], SNR=[100,100])
    signal = Signal(signal_type=signal_type, T=T)
    fig = Signal.plot(signal)
    pyo.plot(fig, filename='signal_test.html', include_mathjax='cdn')