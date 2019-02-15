"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Signal import Signal
from Tool import Particles, Struct

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class FPF(object):
    """
    FPF: Feedback Partilce Filter
        Notation Note:
            N: number of particles
            M: number of observations(y)
        Initialize = FPF(number_of_particles, f_min, f_max, sigma_W, dt, h)
            number_of_particles: integer, number of particles
            f_min: float, minimum frequency in Hz
            f_max: float, maximum frequency in Hz
            sigma_W: 1D list with legnth M, standard deviation of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
        Members = 
            particles: Particles
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
            h_hat: float, filtered observation
        Methods =
            ito_integral(dI, dh): Calculate the ito integral
            calculate_h_hat(): Calculate h for each particle and the average of h (h_hat)
            update(y): Update each particle with new observation data y
        
    """
    def __init__(self, number_of_particles, f_min, f_max, sigma_W, dt, h):
        """Initialize Particles"""
        self.particles = Particles(number_of_particles=number_of_particles, f_min=f_min, f_max=f_max, dt=dt) 
        self.sigma_W = np.reshape(np.array(sigma_W), [len(sigma_W),-1])
        self.dt = dt
        self.h = h
        self.h_hat = self.calculate_h_hat()
    
    def ito_integral(self, dI, dh):
        """Calculate the ito integral"""
        """
            arguments = 
                dI: numpy array with the shape of (N,M), inovation process
                dh: numpy array with the shape of (N,1), the esitmated observation for N particles minus their average
            variables = 
                K: numpy array with the shape of (N,1,M), gain for each particle
                partial_K: numpy array with the shape of (N,1,M), partial_K/partial_theta for each particle
                dI: numpy array with the shape of (N,M,1), inovation process 
        """
        def calculate_K(self, dh):
            """
            Calculate K and partial K with respect to theta
            A*kappa=b

            """
            """
                variables = 
                    K: numpy array with the shape of (N,1,M), gain for each particle
            """
            def calculate_A(self):
                A = np.zeros([2,2])
                c = np.cos(self.particles.theta)
                s = np.sin(self.particles.theta)
                A[0,0] = np.mean(s*s)
                A[0,1] = np.mean(-s*c)
                A[1,0] = A[0,1]
                A[1,1] = np.mean(c*c)
                return A
            
            def calculate_b(self, dh):
                b = np.zeros(2)
                b[0] = np.mean(dh*np.cos(self.particles.theta))
                b[1] = np.mean(dh*np.sin(self.particles.theta))
                return b
            
            A = calculate_A(self)
            b = calculate_b(self, dh)
            K = np.zeros([self.particles.theta.shape[0], self.sigma_W.shape[0]])
            partial_K = np.zeros(K.shape)
            for m, sigma_W in enumerate(self.sigma_W):
                kappa = np.linalg.solve(A,b/sigma_W)
                K[:,m] = np.squeeze(-kappa[0]*np.sin(self.particles.theta) + kappa[1]*np.cos(self.particles.theta))
                partial_K[:,m] = np.squeeze(-kappa[0]*np.cos(self.particles.theta) - kappa[1]*np.sin(self.particles.theta))
            K = np.reshape(K, [K.shape[0], 1, K.shape[1]])
            partial_K = np.reshape(K, K.shape)
            return K, partial_K
        
        K, partial_K = calculate_K(self, dh)
        dI = np.reshape(dI, [dI.shape[0], dI.shape[1], 1])
        return np.reshape(np.matmul(K, dI) + np.matmul(0.5*K*partial_K*self.dt,self.sigma_W**2), [-1,1])

    def calculate_h_hat(self):
        """Calculate h for each particle and the average of h (h_hat)"""
        self.particles.h = self.h(self.particles.amp, self.particles.theta)
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat
    
    def get_h_hat(self):
        return np.squeeze(self.h_hat)

    def update(self, y):
        """Update each particle with new observation data y"""
        """
            argument =
                y: numpy array with the shape of (M,), new observation data
            variables =
                dz: numpy array with the shape of (1,M), integral of y over time period dt
                dI: numpy array with the shape of (N,M), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
        """
        dz = np.reshape(y*self.dt, [1,y.shape[0]])
        dI = dz-np.repeat(0.5*(self.particles.h+self.h_hat)*self.dt, repeats=dz.shape[1], axis=1)
        dtheta = self.particles.omega_bar*self.dt + self.ito_integral(dI=dI, dh=self.particles.h-self.h_hat)
        self.particles.theta += dtheta
        self.particles.amp -= -1*(y-self.particles.h)*2*self.h(1,self.particles.theta)*self.dt          #update amp indivisually
        # self.particles.amp -= -1*(y-self.h_hat)*2*np.mean(self.h(1,self.particles.theta))*self.dt       #update amp together
        self.h_hat = self.calculate_h_hat()
        self.particles.update(theta_error=dtheta-self.particles.omega*self.dt)
        return

    def run(self, y):
        h_hat = np.zeros(y.shape)
        theta = np.zeros([self.particles.N, y.shape[1]])
        amp = np.zeros([self.particles.N, y.shape[1]])
        freq_hat = np.zeros([self.particles.N, y.shape[1]])

        for k in range(y.shape[1]):
            self.update(y[:,k])
            h_hat[:,k] = self.get_h_hat()
            theta[:,k] = self.particles.get_theta()
            amp[:,k] = self.particles.get_amp()
            freq_hat[:,k] = self.particles.get_freq()
        return Struct(h_hat=h_hat, theta=theta, amp=amp, freq_hat=freq_hat)

class Figure(object):
    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):

        if self.fig_property.plot_signal:
            fig, axes = plt.subplots(self.signal.Y.shape[0], 1, figsize=(9,7))
            signal_axes = self.plot_signal(axes)
        
        if self.fig_property.plot_theta:
            fig, ax = plt.subplots(1,1, figsize=(9,7))
            theta_ax = self.plot_theta(ax)
        
        if self.fig_property.plot_omega:
            fig, ax = plt.subplots(1,1, figsize=(9,7))
            omega_ax = self.plot_omega(ax)
        
        if self.fig_property.plot_amp:
            fig, ax = plt.subplots(1,1, figsize=(9,7))
            amp_ax = self.plot_amp(ax)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes):
        try:
            iterator = iter(axes)
        except TypeError:
            ax = axes
            ax.plot(self.signal.t, self.signal.Y[0], label=r'signal')
            ax.plot(self.signal.t, self.filtered_signal.h_hat[0,:], label=r'estimation')
            ax.legend(fontsize=self.fig_property.fontsize-5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_xlabel('time', fontsize=self.fig_property.fontsize)
        else:
            for i, ax in enumerate(axes):
                ax.plot(self.signal.t, self.signal.Y[i], label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[i,:], label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                ax.set_xlabel('time', fontsize=self.fig_property.fontsize)
        return axes

    def plot_theta(self, ax):
        for k in range(self.filtered_signal.theta.shape[0]):
            ax.scatter(self.signal.t, self.filtered_signal.theta[k,:], s=0.5)
        ax.tick_params(labelsize=self.fig_property.fontsize)
        ax.set_xlabel('time', fontsize=self.fig_property.fontsize)
        ax.set_ylabel(r'$\theta$ [rad]', fontsize=self.fig_property.fontsize)
        ax.set_title('Particles', fontsize=self.fig_property.fontsize+2)
        return ax
    
    def plot_omega(self, ax):
        for k in range(self.filtered_signal.freq_hat.shape[0]):
            ax.plot(self.signal.t, self.filtered_signal.freq_hat[k,:]) 
        ax.tick_params(labelsize=self.fig_property.fontsize)
        ax.set_xlabel('time', fontsize=self.fig_property.fontsize)
        ax.set_ylabel('$\omega$ [Hz]', fontsize=self.fig_property.fontsize)
        ax.set_title('Frequency of Particles', fontsize=self.fig_property.fontsize+2)
        
        return ax
    
    def plot_amp(self, ax):
        for k in range(self.filtered_signal.amp.shape[0]):
            ax.plot(self.signal.t, self.filtered_signal.amp[k,:]) 
        ax.tick_params(labelsize=self.fig_property.fontsize)
        ax.set_xlabel('time', fontsize=self.fig_property.fontsize)
        ax.set_ylabel('$A$', fontsize=self.fig_property.fontsize)
        ax.set_title('Amplitude of Particles', fontsize=self.fig_property.fontsize+2)
        return ax
    
    def plot_sync_matrix(self, ax, sync_matrix):
        ax.pcolor(np.arange(1,sync_matrix.shape[0]+1), np.arange(1,sync_matrix.shape[0]+1),\
                    sync_matrix, cmap='gray', vmin=0, vmax=1)
        return ax
    
    def plot_sync_particle(self, ax, sync_particle):
        ax.plot(sync_particle)
        return ax

def h(amp, x):
        return amp*np.cos(x)

if __name__ == "__main__":
    # N states of frequency inputs
    freq = [1.]
    # M-by-N amplitude matrix
    amp = [[10]]
    # N states of state noises
    sigma_B = [0.1]
    # M states of signal noises
    sigma_W = [0.1]

    T = 10.
    dt = 0.001
    signal = Signal(freq=freq, amp=amp, sigma_B=sigma_B, sigma_W=sigma_W, dt=dt, T=T)
    
    N=100
    feedback_particle_filter = FPF(number_of_particles=N, f_min=0.9, f_max=1.1, sigma_W=sigma_W, dt=dt, h=h)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_theta=True, plot_omega=True, plot_amp=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax = figure.plot_sync_matrix(ax, feedback_particle_filter.particles.update_sync_matrix())
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    ax = figure.plot_sync_particle(ax, feedback_particle_filter.particles.check_sync())
    
    plt.show()

