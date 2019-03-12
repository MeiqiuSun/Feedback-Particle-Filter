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
from matplotlib.ticker import PercentFormatter

class FPF(object):
    """FPF: Feedback Partilce Filter
        Notation Note:
            Np: number of particles
            N: number of states (x)
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
    def __init__(self, number_of_particles, model, galerkin, sigma_B, sigma_W, dt):
        self.sigma_B = model.modified_sigma_B(np.array(sigma_B))
        self.sigma_W = model.modified_sigma_W(np.array(sigma_W))
        self.N = len(self.sigma_B)
        self.M = len(self.sigma_W)
        self.dt = dt

        self.particles = Particles(number_of_particles=number_of_particles, \
                                    X0_range=model.X0_range, states_constraints=model.states_constraints, \
                                    M=self.M, dt=self.dt) 
        self.f = model.f
        self.h = model.h
        self.galerkin = galerkin
        self.h_hat = np.zeros(self.M)
    
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
        def calculate_K(self):
            """calculate the optimal feedback gain function K by using Galerkin Approximation to solve the BVP
                Variables = 
                    K: numpy array with the shape of (Np,N,M), gain for each particle
                    partial_K: numpy array with the shape of (Np,N,M,N), partial_K/partial_X for each particle
                    psi: numpy array with the shape of (L,Np), base functions psi
                    grad_psi: numpy array with the shape of (L,N,Np), gradient of base functions psi
                    grad_grad_psi: numpy array with the shape of (L,N,N,Np), gradient of gradient of base functions psi
                    A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
                    b: numpy array with the shape of (L,), b = Exp[normalized_dh*psi] differ from channel to channel (m)
                    c: numpy array with the shape of (L,), the solution to A*c=b
            """
            def calculate_A(self, grad_psi):
                """calculate the A matrix in the Galerkin Approximation in Finite Element Method
                    Arguments = 
                        grad_psi: numpy array with the shape of (L,N,Np), gradient of base functions psi
                    Variables =
                        A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
                """
                A = np.zeros([self.galerkin.L, self.galerkin.L])
                for li in range(self.galerkin.L):
                    for lj in range(li, self.galerkin.L):
                        # A[li,lj] = np.sum(grad_psi[lj]*grad_psi[li])/self.particles.Np
                        A[li,lj] = np.mean(np.sum(grad_psi[lj]*grad_psi[li], axis=0))
                        if not li==lj:
                            A[lj,li] = A[li,lj]
                return A
            
            def calculate_b(self, m, psi):
                """calculate the b vector in the Galerkin Approximation in Finite Element Method
                    Arguments = 
                        m: int, mth of channel (m in [M])
                        psi: numpy array with the shape of (L,Np), base functions psi
                    Variables = 
                        normalized_dh: numpy array with the shape of (Np,), normalized mth channel of filtered observation difference between each particle and mean
                        b = numpy array with the shape of (L,), b = Exp[normalized_dh*psi]
                """
                normalized_dh = (self.particles.h[:,m]-self.h_hat[m])/np.square(self.sigma_W[m])
                b = np.zeros(self.galerkin.L)
                for li in range(self.galerkin.L):
                    b[li] = np.mean(normalized_dh*psi[li])
                return b
            
            K = np.zeros([self.particles.Np, self.N, self.M])
            partial_K = np.zeros([self.particles.Np, self.N, self.M, self.N])
            psi = self.galerkin.psi(X=self.particles.X)
            grad_psi = self.galerkin.grad_psi(X=self.particles.X)
            grad_grad_psi = self.galerkin.grad_grad_psi(X=self.particles.X)
            A = calculate_A(self, grad_psi)
            for m in range(self.M):
                b = calculate_b(self, m, psi)
                c = np.linalg.solve(A,b)
                for l in range(self.galerkin.L):
                    K[:,:,m] += c[l]*np.transpose(grad_psi[l])
                    partial_K[:,:,m,:] += c[l]*np.transpose(grad_grad_psi[l], (2,0,1))
            return K, partial_K
        
        def correction_term(self, K, partial_K):
            """calculate the Wong-Zakai correction term
                correction: numpy array with the shape of (Np,N), correction term from Stratonovich form to Ito form
            """
            correction = np.zeros([self.particles.Np, self.N])
            for ni in range(self.N):
                for nj in range(self.N):
                    correction[:,ni] += np.sum(K[:,nj,:]*partial_K[:,ni,:,nj]*np.square(self.sigma_W), axis=1)
            correction = 0.5*correction
            return correction

        K, partial_K = calculate_K(self)
        dz = np.reshape(np.repeat(dz, repeats=self.particles.Np, axis=0), [self.particles.Np, self.M, 1])
        dI = np.reshape(dI, [dI.shape[0], dI.shape[1], 1])
        
        return np.reshape(np.matmul(K, dI), [self.particles.Np, self.N]) + correction_term(self, K, partial_K)*self.dt
    
    def calculate_h_hat(self):
        """Calculate h for each particle and h_hat by averaging h from all particles"""
        self.particles.h = np.reshape(self.h(self.particles.X), [self.particles.Np, self.M])
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat

    def update(self, y):
        """Update each particle with new observations(y):
            argument =
                y: numpy array with the shape of (M,), new observation data at a fixed time
            variables =
                dz: numpy array with the shape of ( 1,M), integral of y over time period dt
                dI: numpy array with the shape of (Np,M), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
                dX: numpy array with the shape of (Np,N), states increment for each particle
                dU: numpy array with the shape of (Np,N), optimal control input for each particle
        """
        dz = np.reshape(y*self.dt, [1, y.shape[0]])
        dI = dz-np.repeat(0.5*(self.particles.h+self.h_hat)*self.dt, repeats=dz.shape[1], axis=1)
        dX = self.f(self.particles.X)*self.dt + self.sigma_B*np.random.normal(0, np.sqrt(self.dt), size=self.particles.X.shape)
        dU = self.optimal_control(dz, dI)
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
                X: numpy array with the shape of (N,T/dt+1), the states for N particles in time series
            Return =
                filtered_signal: Struct(h_hat, X), filtered observations with information of particles
        """
        h_hat = np.zeros(y.shape)
        X = np.zeros([self.particles.Np, self.N, y.shape[1]])

        for k in range(y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(y[:,k])
            h_hat[:,k] = self.h_hat
            X[:,:,k] = self.particles.X

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
            fig, axes = plt.subplots(self.signal.Y.shape[0], 1, figsize=(9,7))
            signal_axes = self.plot_signal(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_X:
            fig, axes = plt.subplots(self.filtered_signal.X.shape[1],1, figsize=(9,7), sharex=True)
            X_axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)

        if self.fig_property.plot_histogram:
            fig, axes = plt.subplots(self.filtered_signal.X.shape[1],1, figsize=(9,7), sharex=True)
            histogram_axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\npercentage', fontsize=self.fig_property.fontsize)
        
        if self.fig_property.show:
            plt.show()
        
        return

    def plot_signal(self, axes):
        if isiterable(axes):
            for i, ax in enumerate(axes):
                ax.plot(self.signal.t, self.signal.Y[i], color='cyan', label=r'signal')
                ax.plot(self.signal.t, self.filtered_signal.h_hat[i,:], color='magenta', label=r'estimation')
                ax.legend(fontsize=self.fig_property.fontsize-5)
                ax.tick_params(labelsize=self.fig_property.fontsize)
                maximum = np.max(self.signal.Y[i])
                minimum = np.min(self.signal.Y[i])
                signal_range = maximum-minimum
                ax.set_ylim([minimum-signal_range/10, maximum+signal_range/10])
        else:
            ax = axes
            ax.plot(self.signal.t, self.signal.Y[0], color='cyan', label=r'signal')
            ax.plot(self.signal.t, self.filtered_signal.h_hat[0,:], color='magenta', label=r'estimation')
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
                    if np.mod(i,2)==1:
                        state_of_filtered_signal = np.mod(self.filtered_signal.X[k,i,:], 2*np.pi)
                    else:
                        state_of_filtered_signal = self.filtered_signal.X[k,i,:]
                    ax.scatter(self.signal.t, state_of_filtered_signal, s=0.5)
                if np.mod(i,2)==1:
                    state_of_filtered_signal = np.mod(self.filtered_signal.X_average[i,:], 2*np.pi)
                else:
                    state_of_filtered_signal = self.filtered_signal.X_average[i,:]
                # ax.scatter(self.signal.t, state_of_filtered_signal, s=5, color='magenta', label='mean')
                ax.tick_params(labelsize=self.fig_property.fontsize)
                # ax.legend(fontsize=self.fig_property.fontsize-5)
        else:
            ax = axes
            for k in range(self.filtered_signal.X.shape[0]):
                ax.scatter(self.signal.t, self.filtered_signal.X[k,0,:], s=0.5)
            # ax.scatter(self.signal.t, self.filtered_signal.X_average[0,:], s=5, color='magenta', label='mean')
            ax.tick_params(labelsize=self.fig_property.fontsize)
            # ax.legend(fontsize=self.fig_property.fontsize-5)
        return axes

    def plot_histogram(self, axes):
        n_bins = 100
        if isiterable(axes):
            for i, ax in enumerate(axes):
                if np.mod(i,2)==1:
                    state_of_filtered_signal = np.mod(self.filtered_signal.X[:,i,-1], 2*np.pi)
                    # ax.set_xticks(np.arange(0,2*np.pi,5))
                    # ax.set_xticklabels([0, '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
                else:
                    state_of_filtered_signal = self.filtered_signal.X[:,i,-1]
                ax.hist(state_of_filtered_signal, bins=n_bins, orientation='horizontal')
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0]))
                ax.tick_params(labelsize=self.fig_property.fontsize)
        else:
            ax = axes
            ax.hist(self.filtered_signal.X[:, 0,-1], bins=n_bins)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0]))
            ax.tick_params(labelsize=self.fig_property.fontsize)

        return axes

class Model(object):
    def __init__(self):
        self.N = 2
        self.M = 1
        self.X0_range = [[1,2],[0,2*np.pi]]
        self.min_sigma_B = 0.01
        self.min_sigma_W = 0.1

    def modified_sigma_B(self, sigma_B):
        sigma_B = sigma_B.clip(min=self.min_sigma_B)
        sigma_B[1::2] = 0
        return sigma_B
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        return X

    def f(self, X):
        freq_min=0.9
        freq_max=1.1
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = np.linspace(freq_min*2*np.pi, freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        return X[:,0]*np.cos(X[:,1]) 

class Galerkin2(object):    # 2 states (r, theta) 2 base functions [1/2*r^2*cos(theta), 1/2*r^2*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 2          # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ 1/2*np.power(X[:,0],2)*np.cos(X[:,1]), 1/2*np.power(X[:,0],2)*np.sin(X[:,1])]
        return np.array(trial_functions)

    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ np.power(X[:,0],1)*np.cos(X[:,1]), -1/2*np.power(X[:,0],2)*np.sin(X[:,1])],\
                                [ np.power(X[:,0],1)*np.sin(X[:,1]),  1/2*np.power(X[:,0],2)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[                     np.cos(X[:,1]),     -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                      [ -np.power(X[:,0],1)*np.sin(X[:,1]), -1/2*np.power(X[:,0],2)*np.cos(X[:,1])]],\
                                     [[                     np.sin(X[:,1]),      np.power(X[:,0],1)*np.cos(X[:,1])],\
                                      [  np.power(X[:,0],1)*np.cos(X[:,1]), -1/2*np.power(X[:,0],2)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)

class Galerkin1(object):    # 2 states (r, theta) 2 base functions [r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 2       # trial (base) functions
        
    def psi(self, X):
        trial_functions = [ np.power(X[:,0],1)*np.cos(X[:,1]), np.power(X[:,0],1)*np.sin(X[:,1])]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        grad_trial_functions = [[ np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])],\
                                [ np.sin(X[:,1]),  np.power(X[:,0],1)*np.cos(X[:,1])]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        grad_grad_trial_functions = [[[ np.zeros(X[:,0].shape[0]),                     -np.sin(X[:,1])],\
                                      [           -np.sin(X[:,1]), -np.power(X[:,0],1)*np.cos(X[:,1])]],\
                                     [[ np.zeros(X[:,0].shape[0]),                      np.cos(X[:,1])],\
                                      [            np.cos(X[:,1]), -np.power(X[:,0],1)*np.sin(X[:,1])]]]
        return np.array(grad_grad_trial_functions)
        
if __name__ == "__main__":

    T = 10
    sampling_rate = 160 # Hz
    dt = 1./sampling_rate
    signal_type = Sinusoidal(dt)
    signal = Signal(signal_type=signal_type, T=T)
    
    Np = 1000
    feedback_particle_filter = FPF(number_of_particles=Np, model=Model(), galerkin=Galerkin1(), sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, plot_histogram=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figure.plot()

    plt.show()

