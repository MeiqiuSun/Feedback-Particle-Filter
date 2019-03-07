"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Signal import Signal, Sinusoidal
from Tool import Particles, Struct, isiterable

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class FPF(object):
    """FPF: Feedback Partilce Filter
        Notation Note:
            Np: number of particles
            M: number of observations(y)
        Initialize = FPF(number_of_particles, f_min, f_max, sigma_W, dt, h)
            number_of_particles: integer, number of particles
            f_min: float, minimum frequency in Hz
            f_max: float, maximum frequency in Hz
            sigma_W: 1D list of float with legnth M, standard deviations of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
            indep_amp_update: bool, if update amplitude independently 
        Members = 
            particles: Particles
            sigma_W: numpy array with the shape of (M,1), standard deviation of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
            h_hat: numpy array with the shape of (M,), filtered observations
            indep_amp_update: bool, if update amplitude independently 
        Methods =
            correction(dI, dh): Correct the theta prediction with new observations(y)
            calculate_h_hat(): Calculate h for each particle and h_hat by averaging h from all particles
            calculate_omega(): Calculate average omega from all particles
            calculate_amp(): Calculate average amplitude form all particles
            get_freq(): return average frequency (Hz) 
            get_amp(): return average amplitude 
            update(y): Update each particle with new observations(y)
            run(y): Run FPF with time series observations(y)
    """

    def __init__(self, number_of_particles, X0_range, states_constraints, sigma_B, sigma_W, min_sigma_B, min_sigma_W, f, h, dt, galerkin):
        self.particles = Particles(number_of_particles=number_of_particles, X0_range=X0_range, states_constraints=states_constraints, M=len(sigma_W), dt=dt) 
        self.sigma_B = np.array(sigma_B).clip(min=min_sigma_B)
        self.sigma_B[1] = 0
        self.sigma_W = np.array(sigma_W).clip(min=min_sigma_W)
        self.dt = dt
        self.f = f
        self.h = h
        self.galerkin = galerkin
        self.h_hat = np.zeros(len(sigma_W))
    
    def optimal_control(self, dz, dI):
        """calculate the optimal control with new observations(y):
            Arguments = 
                dz: numpy array with the shape of ( 1,M), integral of y over time period dt
                dI: numpy array with the shape of (Np,M), inovation process
                dh: numpy array with the shape of (Np,M), the esitmated observation for Np particles minus their average
            Variables = 
                K: numpy array with the shape of (Np,N,M), gain for each particle
                partial_K: numpy array with the shape of (Np,N,M,N), partial_K/partial_X for each particle
                dI: numpy array with the shape of (Np,M,1), inovation process 
        """
        def calculate_K(self, dh):
            
            def calculate_A(self, grad_psi):
                A = np.zeros([self.galerkin.L, self.galerkin.L])
                for li in range(self.galerkin.L):
                    for lj in range(li, self.galerkin.L):
                        A[li,lj] = np.mean(grad_psi[lj]*grad_psi[li])
                        if not li==lj:
                            A[lj,li] = A[li,lj]
                return A
            
            def calculate_b(self, normalized_dh, psi):
                b = np.zeros(self.galerkin.L)
                for li in range(self.galerkin.L):
                    b[li] = np.mean(normalized_dh*psi[li])
                return b
            
            K = np.zeros([self.particles.Np, self.sigma_B.shape[0], self.sigma_W.shape[0]])
            partial_K = np.zeros([self.particles.Np, self.sigma_B.shape[0], self.sigma_W.shape[0], self.sigma_B.shape[0]])
            psi = self.galerkin.psi(l=0, X=self.particles.X)
            grad_psi = self.galerkin.grad_psi(l=0, X=self.particles.X)
            grad_grad_psi = self.galerkin.grad_grad_psi(l=0, X=self.particles.X)
            A = calculate_A(self, grad_psi)
            for m in range(self.sigma_W.shape[0]):
                b = calculate_b(self, dh[:,m]/self.sigma_W[m], psi)
                c = np.linalg.solve(A,b)
                for l in range(self.galerkin.L):
                    K[:,:,m] += c[l]*np.transpose(grad_psi[l])
                    partial_K[:,:,m,:] += c[l]*np.transpose(grad_grad_psi[l], (2,0,1))
            return K, partial_K
        
        def calculate_Omega(self, K, partial_K):
            Omega = np.zeros([self.particles.Np, self.sigma_B.shape[0]])
            for ni in range(self.sigma_B.shape[0]):
                for nj in range(self.sigma_B.shape[0]):
                    Omega[:,ni] += np.sum(K[:,nj,:]*partial_K[:,ni,:,nj]*self.sigma_W, axis=1)
            Omega = 0.5*Omega
            return Omega

        K, partial_K = calculate_K(self, self.particles.h-self.h_hat)
        dz = np.reshape(np.repeat(dz, repeats=self.particles.Np, axis=0), [self.particles.Np, self.sigma_W.shape[0], 1])
        dI = np.reshape(dI, [dI.shape[0], dI.shape[1], 1])
        Omega = calculate_Omega(self, K, partial_K)
        return np.reshape(np.matmul(K, dI), [self.particles.Np, self.sigma_B.shape[0]]) + Omega*self.dt
    
    def calculate_h_hat(self):
        """Calculate h for each particle and h_hat by averaging h from all particles"""
        self.particles.h = np.reshape(self.h(self.particles.X), [self.particles.Np, self.sigma_W.shape[0]])
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat

    def update(self, y):
        """Update each particle with new observations(y):
            argument =
                y: numpy array with the shape of (M,), new observation data
            variables =
                dz: numpy array with the shape of ( 1,M), integral of y over time period dt
                dI: numpy array with the shape of (Np,M), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
        """
        dz = np.reshape(y*self.dt, [1, y.shape[0]])
        dI = dz-np.repeat(0.5*(self.particles.h+self.h_hat)*self.dt, repeats=dz.shape[1], axis=1)
        dX = self.f(self.particles.X)*self.dt + self.sigma_B*np.random.normal(0, np.sqrt(self.dt), size=self.particles.X.shape)
        dU = self.optimal_control(dz, dI)
        if np.isnan(dU[0,0]):
            quit()
        self.particles.X += dX+dU
        self.h_hat = self.calculate_h_hat()
        self.particles.update()
        return

    def run(self, y):
        """Run FPF with time series observations(y):
            Arguments = 
                y: numpy array with the shape of (M,T/dt+1), observations in time series
            Variables = 
                h_hat: numpy array with the shape of (M,T/dt+1), filtered observations in time series
                theta: numpy array with the shape of (N,T/dt+1), the angles (rad) for N particles in time series
                freq: numpy array with the shape of (N,T/dt+1), the updated frequency (Hz) for N particles in time series
                amp: numpy array with the shape of (N,T/dt+1), the amplitude for N particles in time series
            Return =
                filtered_signal: Struct(h_hat, theta, freq, amp), filtered observations with information of particles
        """
        h_hat = np.zeros(y.shape)
        X = np.zeros([self.particles.Np, self.sigma_B.shape[0], y.shape[1]])

        for k in range(y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(y[:,k])
            h_hat[:,k] = self.h_hat
            X[:,:,k] = self.particles.X
            # if k==3:
            #     quit()

        return Struct(h_hat=h_hat, X=X)

class Figure(object):
    """Figure: Figures for Feedback Partilce Filter
        Initialize = Figure(fig_property, signal, filtered_signal)
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Members = 
            fig_property: Sturct, properties for figures and plot flags
            signal: Signal, observations
            filtered_signal: Struct, filtered observations with information of particles
        Methods =
            plot(): plot figures
            plot_signal(axes): plot a figure with observations and filtered observations together, each of axes represents a time series of observation
            plot_theta(ax): plot theta for each particles
            plot_freq(ax): plot frequency for each particles and the average frequency
            plot_amp(ax): plot amplitude for each particles and the average amplitude
            plot_sync_matrix(ax, sync_matrix): plot synchronized matrix
            plot_sync_particles(ax, sync_particles): plot synchronized particles
    """

    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
    
    def plot(self):
        if self.fig_property.plot_signal:
            fig, axes = plt.subplots(self.signal.Y.shape[0], 1, figsize=(9,7), sharex=True)
            signal_axes = self.plot_signal(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('time [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_X:
            fig, axes = plt.subplots(self.filtered_signal.X.shape[1],1, figsize=(9,7))
            X_axes = self.plot_X(axes)

        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes):
        if isiterable(axes):
            for i, ax in enumerate(axes):
                ax.plot(self.signal.t, self.signal.Y[i], label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[i,:], label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                maximum = np.max(self.signal.Y[i])
                minimum = np.min(self.signal.Y[i])
                signal_range = maximum-minimum
                ax.set_ylim([minimum-signal_range/10, maximum+signal_range/10])
        else:
            ax = axes
            ax.plot(self.signal.t, self.signal.Y[0], label=r'signal')
            ax.plot(self.signal.t, self.filtered_signal.h_hat[0,:], label=r'estimation')
            ax.legend(fontsize=self.fig_property.fontsize-5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            maximum = np.max(self.signal.Y[0])
            minimum = np.min(self.signal.Y[0])
            signal_range = maximum-minimum
            ax.set_ylim([minimum-signal_range/10, maximum+signal_range/10])
        return axes

    def plot_X(self, axes):
        if isiterable(axes):
            for i, ax in enumerate(axes):
                for k in range(self.filtered_signal.X.shape[0]):
                    ax.scatter(self.signal.t, np.mod(self.filtered_signal.X[k,i,:], 2*np.pi), s=0.5)
                # ax.plot(self.signal.t, np.mod(np.mean(self.filtered_signal.X[:,i,:], axis=0), 2*np.pi), label='mean')
                ax.tick_params(labelsize=self.fig_property.fontsize)
                # ax.legend(fontsize=self.fig_property.fontsize-5)
        else:
            ax = axes
            for k in range(self.filtered_signal.X.shape[0]):
                ax.scatter(self.signal.t, self.filtered_signal.X[k,0,:], s=0.5)
            # ax.plot(self.signal.t, np.mean(self.filtered_signal.X[:,i,:], axis=0), label='mean')
            ax.tick_params(labelsize=self.fig_property.fontsize)
            # ax.legend(fontsize=self.fig_property.fontsize-5)
        return axes
    
    def plot_sync_matrix(self, ax, sync_matrix):
        ax.pcolor(np.arange(1,sync_matrix.shape[0]+1), np.arange(1,sync_matrix.shape[0]+1),\
                    sync_matrix, cmap='gray', vmin=0, vmax=1)
        return ax
    
    def plot_sync_particles(self, ax, sync_particles):
        ax.plot(sync_particles)
        return ax

def f(X):
    freq_min=0.9
    freq_max=1.1
    dXdt = np.zeros(X.shape)
    # dXdt[:,0] = np.linspace(freq_min, freq_max, X.shape[0])
    dXdt[:,1] = np.linspace(freq_min*2*np.pi, freq_max*2*np.pi, X.shape[0])
    return dXdt

def h(X):
    return X[:,0]*np.cos(X[:,1]) 

# 2 states (r, theta) 2 base functions [1/2*r^2*cos(theta), 1/2*r^2*sin(theta)]
class Galerkin(object):
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 2       # trial (base) functions
        
    def psi(self, l, X):
        trial_functions = [ 1/2*np.power(X[:,0],2)*np.cos(X[:,1]), 1/2*np.power(X[:,0],2)*np.sin(X[:,1])]
        if l==0:
            return np.array(trial_functions)
        else:
            return trial_functions[l-1]

    # gradient of trial (base) functions
    def grad_psi(self, l, X):
        grad_trial_functions = [[ np.power(X[:,0],1)*np.cos(X[:,1]), -1/2*np.power(X[:,0],2)*X[:,0]*np.sin(X[:,1])],\
                                [ np.power(X[:,0],1)*np.sin(X[:,1]),  1/2*np.power(X[:,0],2)*X[:,0]*np.cos(X[:,1])]]
        if l==0:
            return np.array(grad_trial_functions)
        else:
            return np.array(grad_trial_functions[l-1])
    
    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, l, X):
        grad_grad_trial_functions = [[[                     np.cos(X[:,1]),     -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                      [ -np.power(X[:,0],1)*np.sin(X[:,1]), -1/2*np.power(X[:,0],2)*np.cos(X[:,1])]],\
                                     [[                     np.sin(X[:,1]),      np.power(X[:,0],1)*np.cos(X[:,1])],\
                                      [  np.power(X[:,0],1)*np.cos(X[:,1]), -1/2*np.power(X[:,0],1)*np.sin(X[:,1])]]]
        if l==0:
            return np.array(grad_grad_trial_functions)
        else:
            return np.array(grad_grad_trial_functions[l-1])

def states_constraints(X):
    X[:,0] = X[:,0].clip(min=0)
    # X[:,1] = np.mod(X[:,1], 2*np.pi)
    return X

if __name__ == "__main__":

    T = 100
    sampling_rate = 160 # Hz
    dt = 1./sampling_rate
    signal_type = Sinusoidal(dt)
    signal = Signal(f=signal_type.f, h=signal_type.h, sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, X0=signal_type.X0, dt=dt, T=T)
    
    N=10
    galerkin = Galerkin()
    
    min_sigma_B = 0.05
    min_sigma_W = 0.001
    X0_range = [[0.5,1.5],[0,2*np.pi]]
    feedback_particle_filter = FPF(number_of_particles=N, X0_range=X0_range, states_constraints=states_constraints, \
                                    sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, min_sigma_B=min_sigma_B, min_sigma_W=min_sigma_W, \
                                    f=f, h=h, dt=dt, galerkin=galerkin)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_matrix(ax, feedback_particle_filter.particles.update_sync_matrix())
    
    # fig, ax = plt.subplots(1,1, figsize=(7,7))
    # ax = figure.plot_sync_particles(ax, feedback_particle_filter.particles.check_sync())
    
    plt.show()

