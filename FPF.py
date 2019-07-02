"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from abc import ABC, abstractmethod

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from Signal import Signal, Sinusoidal
from Tool import Struct, isiterable, find_limits


class Particles(object):
    """Particles:
        Notation Note:
            d: dimenstion of states
        Initialize = Particles(number_of_particles, X0_range, states_constraints, m)
            number_of_particles: integer, number of particles
            X0_range: 2D list of float with length d-by-2, range of initial states
            states_constraints: function, constraints on states
            m: integer, number of outputs
        Members =
            Np: integer, number of particles
            X: numpy array with the shape of (Np,d), the estimated states for Np particles
            h: numpy array with the shape of (Np,m), the esitmated observations for Np particles
            states_constraints: function, constraints on states
        Methods =
            update(): use states_constraints function to update states in all particles
            get_X(): return X (if the dimenstion of states is 1, then the shape of returned array is (Np,), otherwise (Np, d))
    """

    def __init__(self, number_of_particles, X0_range, states_constraints, m):
        self.Np = int(number_of_particles)
        self.X = np.zeros([self.Np, len(X0_range)])
        for d in range(len(X0_range)):
            self.X[:,d] = np.linspace(X0_range[d][0], X0_range[d][1], self.Np)
        self.h = np.zeros([self.Np, m])
        self.states_constraints = states_constraints

    def update(self, dX):
        self.X = self.states_constraints(self.X+dX)
        return 

    def get_X(self):
        return np.squeeze(self.X)

class Model(ABC):
    def __init__(self, m, d, X0_range, states, states_label, sigma_V, sigma_W):
        self.m = int(m)
        self.d = int(d)
        self.X0_range = np.array(X0_range)
        self.states = states
        self.states_label = states_label
        self.sigma_V = np.array(sigma_V)
        self.sigma_W = np.array(sigma_W)
    
    @abstractmethod
    def states_constraints(self, X):
        return X
    
    @abstractmethod
    def f(self, X):
        dXdt = np.zeros(X.shape)
        return dXdt

    @abstractmethod
    def h(self, X):
        Y = np.zeros((X.shape[0], self.m))
        return Y

class Galerkin(ABC):
    def __init__(self, states, m, L):
        self.states = ['self.'+state for state in states]
        self.d = len(states)
        self.m = int(m)
        self.L = int(L)
        self.c = np.zeros((self.m, self.L))
    
    def split(self, X):
        for d in range(self.d):
            exec(self.states[d]+' = X[:,{}]'.format(d))

    @abstractmethod
    def psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method psi(self, X)")
    
    
    @abstractmethod
    def grad_psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method grad_psi(self, X)")

    
    @abstractmethod
    def grad_grad_psi(self, X):
        raise NotImplementedError("Subclasses of Galerkin must set an instance method grad_grad_psi(self, X)")    
    
    def calculate_K(self, X, normalized_dh):
        """calculate the optimal feedback gain function K by using Galerkin Approximation to solve the BVP
            Arguments = 
                X: numpy array with the shape of (Np, d), the estimated states for Np particles
                normalized_dh: numpy array with the shape of (Np,m), normalized filtered observation difference between each particle and mean
            Variables = 
                K: numpy array with the shape of (Np,d,m), gain for each particle
                partial_K: numpy array with the shape of (Np,d,m,d), partial_K/partial_X for each particle
                psi: numpy array with the shape of (L,Np), base functions psi
                grad_psi: numpy array with the shape of (L,d,Np), gradient of base functions psi
                grad_grad_psi: numpy array with the shape of (L,d,d,Np), gradient of gradient of base functions psi
                A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
                b: numpy array with the shape of (m,L), b = Exp[normalized_dh*psi] differ from channel to channel (m)
                c: numpy array with the shape of (m,L), the solution to A*c=b
        """
        def calculate_A(self, grad_psi):
            """calculate the A matrix in the Galerkin Approximation in Finite Element Method
                Arguments = 
                    grad_psi: numpy array with the shape of (L,d,Np), gradient of base functions psi
                Variables =
                    A: numpy array with the shape of (L,L), A_(i,j) = Exp[grad_psi_j dot grad_psi_i]
            """
            A = np.zeros([self.L, self.L])
            for li in range(self.L):
                for lj in range(li, self.L):
                    # A[li,lj] = np.sum(grad_psi[lj]*grad_psi[li])/self.particles.Np
                    A[li,lj] = np.mean(np.sum(grad_psi[lj]*grad_psi[li], axis=0))
                    if not li==lj:
                        A[lj,li] = A[li,lj]
            # print(np.linalg.det(A))
            if np.linalg.det(A)<=0.01:
                A += np.diag(0.01*np.random.randn(self.L))
            # np.set_printoptions(precision=1)
            # print(A)
            return A
        
        def calculate_b(self, normalized_dh, psi):
            """calculate the b vector in the Galerkin Approximation in Finite Element Method
                Arguments = 
                    normalized_dh: numpy array with the shape of (Np,), normalized mth channel of filtered observation difference between each particle and mean
                    psi: numpy array with the shape of (L,Np), base functions psi
                Variables = 
                    b = numpy array with the shape of (L,), b = Exp[normalized_dh*psi]
            """
            b = np.zeros(self.L)
            for l in range(self.L):
                b[l] = np.mean(normalized_dh*psi[l])
            return b
        
        K = np.zeros((X.shape[0], self.d, self.m))
        partial_K = np.zeros((X.shape[0], self.d, self.m, self.d))
        grad_psi = self.grad_psi(X)            
        grad_grad_psi = self.grad_grad_psi(X)
        A = calculate_A(self, grad_psi)
        b = np.zeros((self.m, self.L))
        for m in range(self.m):
            b[m] = calculate_b(self, normalized_dh[:,m], self.psi(X))
            self.c[m] = np.linalg.solve(A,b[m])
            for l in range(self.L):
                K[:,:,m] += self.c[m,l]*np.transpose(grad_psi[l])
                partial_K[:,:,m,:] += self.c[m,l]*np.transpose(grad_grad_psi[l], (2,0,1))
        return K, partial_K

class FPF(object):
    """FPF: Feedback Partilce Filter
        Notation Note:
            Np: number of particles
        Initialize = FPF(number_of_particles, model, galerkin, dt)
            number_of_particles: integer, number of particles
            model: class Model, f, h and other paratemers are defined at here
            galerkin: class Galerkin, base functions are defined at here
            dt: float, size of time step in sec
        Members =
            d: number of states(X)
            m: number of observations(Y)
            sigma_V: numpy array with the shape of (d,), standard deviations of process noise(X)
            sigma_W: numpy array with the shape of (m,), standard deviations of noise of observations(Y)
            f: function, state dynamics
            h: function, estimated observation dynamics
            dt: float, size of time step in sec
            particles: class Particles,
            galerkin: class Galerkin, base functions are defined at here
            c: numpy array with the shape of (m, L), 
            h_hat: numpy array with the shape of (m,), filtered observations
        Methods =
            optimal_control(dZ, dI): calculate the optimal control with new observations(Y)
            calculate_h_hat(): Calculate h for each particle and h_hat by averaging h from all particles
            update(Y): Update each particle with new observations(Y)
            run(Y): Run FPF with time series observations(Y)
    """
    def __init__(self, number_of_particles, model, galerkin, dt):
        self.d = model.d
        self.m = model.m
        self.sigma_V = model.sigma_V
        self.sigma_W = model.sigma_W
        self.f = model.f
        self.h = model.h
        self.dt = dt

        self.particles = Particles(number_of_particles=number_of_particles, \
                                    X0_range=model.X0_range, states_constraints=model.states_constraints, \
                                    m=self.m) 
        self.galerkin = galerkin
        self.h_hat = np.zeros(self.m)
    
    def optimal_control(self, dI, dZ):
        """calculate the optimal control with new observations(Y):
            Arguments = 
                dI: numpy array with the shape of (Np,m), inovation process
            Variables = 
                K: numpy array with the shape of (Np,d,m), gain for each particle
                partial_K: numpy array with the shape of (Np,d,m,d), partial_K/partial_X for each particle
        """
        K, partial_K = self.galerkin.calculate_K(self.particles.X, (self.particles.h-self.h_hat)/np.square(self.sigma_W))
        # return np.einsum('inm,m->in',K,dZ[0,:]) - np.einsum('inm,m->in',K,self.h_hat)*self.dt
        return np.einsum('idm,im->id',K,dI) + 0.5*np.einsum('ijm,idmj->id',K,np.einsum('idmj,m->idmj',partial_K,np.square(self.sigma_W)))*self.dt
    
    def calculate_h_hat(self):
        """Calculate h for each particle and h_hat by averaging h from all particles"""
        self.particles.h = np.reshape(self.h(self.particles.X), [self.particles.Np, self.m])
        h_hat = np.mean(self.particles.h, axis=0)
        return h_hat

    def update(self, Y):
        """Update each particle with new observations(Y):
            argument =
                Y: numpy array with the shape of (m,), new observation data at a fixed time
            variables =
                dZ: numpy array with the shape of ( 1,m), integral of y over time period dt
                dI: numpy array with the shape of (Np,m), inovation process dI = dz-0.5*(h+h_hat)*dt for each particle
                dX: numpy array with the shape of (Np,d), states increment for each particle
                dU: numpy array with the shape of (Np,d), optimal control input for each particle
        """
        dZ = np.reshape(Y*self.dt, [1, Y.shape[0]])
        dI = dZ-0.5*(self.particles.h+self.h_hat)*self.dt
        dX = self.f(self.particles.X)*self.dt + self.sigma_V*np.random.normal(0, np.sqrt(self.dt), size=self.particles.X.shape)
        dU = self.optimal_control(dI, dZ)
        self.particles.update(dX+dU)
        self.h_hat = self.calculate_h_hat()
        return dU

    def run(self, Y, show_time=False):
        """Run FPF with time series observations(Y):
            Arguments = 
                Y: numpy array with the shape of (m,T/dt+1), observations in time series
            Variables = 
                h_hat: numpy array with the shape of (m,T/dt+1), filtered observations in time series
                X: numpy array with the shape of (d,T/dt+1), the states for Np particles in time series
            Return =
                filtered_signal: Struct(h_hat, X), filtered observations with information of particles
        """
        h_hat = np.zeros(Y.shape)
        error = np.zeros(Y.shape)
        h = np.zeros([self.particles.Np, self.m, Y.shape[1]])
        X = np.zeros([self.particles.Np, self.d, Y.shape[1]])
        dU = np.zeros([self.particles.Np, self.d, Y.shape[1]])
        c = np.zeros([self.m, self.galerkin.L, Y.shape[1]]) 
        X_mean = np.zeros([self.d, Y.shape[1]])
        for k in range(Y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            dU[:,:,k] = self.update(Y[:,k])
            h_hat[:,k] = self.h_hat
            h[:,:,k] = self.particles.h
            X[:,:,k] = self.particles.X
            X_mean[:,k] = np.mean(self.particles.X, axis=0)
            c[:,:,k] = self.galerkin.c
        # for m in range(y.shape[0]):
        #     L2_error[m,:] = np.sqrt(np.cumsum(np.square(y[m]-h_hat[m])*self.dt)/(self.dt*(y.shape[1]-1)))/self.sigma_W[m]
        for m in range(self.m):
            error[m,:] = h_hat[m,:]-Y[m,:]
        return Struct(h_hat=h_hat, error=error, h=h, X=X, X_mean=X_mean, dU=dU, c=c)

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
            plot_signal(axes, m): plot a figure with observations and filtered observations together, each of axes represents a time series of observation
            
    """
    def __init__(self, fig_property, signal, filtered_signal):
        self.fig_property = fig_property
        self.signal = signal
        self.filtered_signal = filtered_signal
        self.figs = {'fontsize':self.fig_property.fontsize}
    
    def plot(self):

        if ('plot_signal' in self.fig_property.__dict__) and self.fig_property:
            print("plot_signal")
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(8,8))
                ax2 = plt.subplot2grid(shape=(4,1),loc=(3,0))
                ax1 = plt.subplot2grid(shape=(4,1),loc=(0,0), rowspan=3, sharex=ax2)
                axes = self.plot_signal([ax1, ax2], m)
                ax2.axhline(y=0, color='gray', linestyle='--')
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
                plt.title('m={}'.format(m+1), fontsize=self.fig_property.fontsize+2)
                self.figs['signal_{}'.format(m+1)] = Struct(fig=fig, axes=axes)

        if ('plot_X' in self.fig_property.__dict__) and self.fig_property.plot_X:
            print("plot_X")
            nrow = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow, 1, figsize=(8,4*nrow+4), sharex=True)
            axes = [axes] if nrow==1 else axes
            axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
            self.figs['X'] = Struct(fig=fig, axes=axes)

        if ('plot_histogram' in self.fig_property.__dict__) and self.fig_property.plot_histogram:
            print("plot_histogram")
            nrow = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow ,1, figsize=(8,4*nrow+4), sharex=True)
            axes = [axes] if nrow==1 else axes
            axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\nhistogram [%]', fontsize=self.fig_property.fontsize)
            self.figs['histogram'] = Struct(fig=fig, axes=axes)
        
        if ('plot_c' in self.fig_property.__dict__) and self.fig_property.plot_c:
            print("plot_c")
            for m in range(self.signal.Y.shape[0]):
                nrow = self.filtered_signal.c.shape[1]
                fig, axes = plt.subplots(nrow, 1, figsize=(8,4*nrow+4), sharex=True)
                axes = [axes] if nrow==1 else axes
                axes = self.plot_c(axes, m)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
                plt.title('m={}'.format(m+1), fontsize=self.fig_property.fontsize+2)
                self.figs['c,m={}'.format(m+1)] = Struct(fig=fig, axes=axes)
        
        return self.figs

    def plot_signal(self, axes, m):
        axes[0].plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal $Y_t$')
        axes[0].plot(self.signal.t, self.filtered_signal.h_hat[m], color='magenta', label=r'estimation $\hat h_t$')
        axes[0].legend(fontsize=self.fig_property.fontsize-5, loc='lower right', ncol=2)
        axes[0].tick_params(labelsize=self.fig_property.fontsize)
        axes[0].set_ylim(find_limits(self.signal.Y))
        
        axes[1].plot(self.signal.t,self.filtered_signal.error[m], color='magenta')
        axes[1].tick_params(labelsize=self.fig_property.fontsize)
        axes[1].set_ylim(find_limits(self.filtered_signal.error[m]))
        # axes[1].set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)
        axes[1].set_ylabel('error', fontsize=self.fig_property.fontsize)
        return axes

    def modified(self, signal, n):
        if not 'restrictions' in self.fig_property.keys():
            return signal
        else:
            signal = self.fig_property.restrictions.scaling[n]*signal
            if not np.isnan(self.fig_property.restrictions.mod[n]):
                signal = np.mod(signal, self.fig_property.restrictions.mod[n])
        return signal

    def set_limits(self, signal, n=np.nan, center=False):
        limits = find_limits(signal)
        if not 'restrictions' in self.fig_property.keys():
            return limits
        if not np.isnan(n):
            if not np.isnan(self.fig_property.restrictions.mod[n]):
                limits = find_limits(np.array([0, self.fig_property.restrictions.mod[n]]))
            if self.fig_property.restrictions.zero[n]:
                limits = find_limits(np.append(signal,0))
        if center or (self.fig_property.restrictions.center[n] if not np.isnan(n) else False):
            maximum = np.max(np.absolute(limits))
            limits = [-maximum, maximum]
        return limits

    def plot_X(self, axes):
        Np = self.filtered_signal.X.shape[0]
        if 'particles_ratio' in self.fig_property.__dict__ :
            sampling_number = np.random.choice(Np, size=int(Np*self.fig_property.particles_ratio), replace=False)
        else :
            sampling_number = np.array(range(Np))

        def plot_Xn(self, ax, sampling_number, n=0):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[:,n,:], n)
            for i in range(sampling_number.shape[0]):
                ax.scatter(self.signal.t, state_of_filtered_signal[sampling_number[i]], s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, n=n))
            return ax

        for n, ax in enumerate(axes):
            plot_Xn(self, ax, sampling_number, n)
        return axes

    def plot_histogram(self, axes):

        def plot_histogram_Xn(self, ax, bins, n=0, k=-1):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[:,n,:], n)
            ax.hist(state_of_filtered_signal[:,k], bins=bins, color='blue', orientation='horizontal')
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0], symbol=''))
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, n=n))
            return ax
        
        if 'n_bins' in self.fig_property.__dict__ :
            bins = self.fig_property.n_bins
        else :
            bins = 100

        for n, ax in enumerate(axes):
            plot_histogram_Xn(self, ax, bins, n)
        return axes

    def plot_c(self, axes, m):

        def plot_cl(self, ax, m, l=0):
            ax.plot(self.signal.t, self.filtered_signal.c[m,l,:], color='magenta')
            ax.plot(self.signal.t, np.zeros(self.signal.t.shape), linestyle='--', color='gray', alpha=0.3)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(self.filtered_signal.c[m,:,int(self.signal.t.shape[0]/2):], center=True))
            ax.set_ylabel('$c_{}$'.format(l+1), fontsize=self.fig_property.fontsize, rotation=0)
            return ax

        for l, ax in enumerate(axes):
            plot_cl(self, ax, m, l)

        return axes

class Model1(Model):
    def __init__(self, sigma_V, sigma_W):
        self.freq_min=0.9
        self.freq_max=1.1
        self.sigma_V_min = 0.01
        self.sigma_W_min = 0.1
        Model.__init__(self, m=1, d=2, X0_range=[[1,2],[0,2*np.pi]], 
                        states=['r','theta'], states_label=[r'$r$',r'$\theta$'], 
                        sigma_V=self.modified_sigma_V(sigma_V), 
                        sigma_W=self.modified_sigma_W(sigma_W))

    def modified_sigma_V(self, sigma_V):
        sigma_V = np.array(sigma_V).clip(min=self.sigma_V_min)
        sigma_V[1::2] = 0
        return sigma_V
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = np.array(sigma_W).clip(min=self.sigma_W_min)
        return sigma_W

    def states_constraints(self, X):
        # X[:,1] = np.mod(X[:,1], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,1] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
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

class Galerkin1(Galerkin):    # 2 states (r, theta) 2 base functions [r*cos(theta), r*sin(theta)]
    # Galerkin approximation in finite element method
    def __init__(self, states, m):
        # L: int, number of trial (base) functions
        Galerkin.__init__(self, states, m, L=2)
        
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
    model = Model1(sigma_V=signal_type.sigma_V, sigma_W=signal_type.sigma_W)
    galerkin = Galerkin1(model.states, model.m)
    feedback_particle_filter = FPF(number_of_particles=Np, model=model, galerkin=galerkin, dt=signal_type.dt)
    filtered_signal = feedback_particle_filter.run(signal.Y)

    fontsize = 20
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_histogram=True, plot_c=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figs = figure.plot()

    plt.show()

