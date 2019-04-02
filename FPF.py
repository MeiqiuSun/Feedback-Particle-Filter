"""
Created on Wed Jan. 23, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from Signal import Signal, Sinusoidal
from Tool import Particles, Struct, isiterable, find_limits

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
        self.c = np.zeros([self.M, self.galerkin.L])
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
                    b: numpy array with the shape of (M,L), b = Exp[normalized_dh*psi] differ from channel to channel (m)
                    c: numpy array with the shape of (M,L), the solution to A*c=b
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
                if np.linalg.det(A)==0:
                    A += np.diag(0.01*np.random.randn(self.galerkin.L))
                # np.set_printoptions(precision=1)
                # print(A)
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
            b = np.zeros([self.M, self.galerkin.L])
            c = np.zeros([self.M, self.galerkin.L])
            for m in range(self.M):
                b[m] = calculate_b(self, m, psi)
                c[m] = np.linalg.solve(A,b[m])
                for l in range(self.galerkin.L):
                    K[:,:,m] += c[m,l]*np.transpose(grad_psi[l])
                    partial_K[:,:,m,:] += c[m,l]*np.transpose(grad_grad_psi[l], (2,0,1))
            self.c = c
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
        return dU

    def run(self, y, show_time=False):
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
        L2_error = np.zeros(y.shape)
        h = np.zeros([self.particles.Np, self.M, y.shape[1]])
        X = np.zeros([self.particles.Np, self.N, y.shape[1]])
        dU = np.zeros([self.particles.Np, self.N, y.shape[1]])
        c = np.zeros([self.M, self.galerkin.L, y.shape[1]]) 
        X_mean = np.zeros([self.N, y.shape[1]])
        for k in range(y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            dU[:,:,k] = self.update(y[:,k])
            h_hat[:,k] = self.h_hat
            h[:,:,k] = self.particles.h
            X[:,:,k] = self.particles.X
            X_mean[:,k] = np.mean(self.particles.X, axis=0)
            c[:,:,k] = self.c
        # for m in range(y.shape[0]):
        #     L2_error[m,:] = np.sqrt(np.cumsum(np.square(y[m]-h_hat[m])*self.dt)/(self.dt*(y.shape[1]-1)))/self.sigma_W[m]
        for m in range(y.shape[0]):
            L2_error[m,:] = y[m]-h_hat[m]
        return Struct(h_hat=h_hat, L2_error=L2_error, h=h, X=X, X_mean=X_mean, dU=dU, c=c)

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

        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(8,8))
                ax2 = plt.subplot2grid((4,1),(3,0))
                ax1 = plt.subplot2grid((4,1),(0,0), rowspan=3, sharex=ax2)
                axes = self.plot_signal([ax1, ax2], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
                plt.title('m={}'.format(m+1), fontsize=self.fig_property.fontsize+2)
                self.figs['signal_{}'.format(m+1)] = Struct(fig=fig, axes=axes)

        if self.fig_property.plot_X:
            nrow = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow, 1, figsize=(8,4*nrow+4), sharex=True)
            axes = [axes] if nrow==1 else axes
            axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
            self.figs['X'] = Struct(fig=fig, axes=axes)

        if self.fig_property.plot_histogram:
            nrow = self.filtered_signal.X.shape[1]
            fig, axes = plt.subplots(nrow ,1, figsize=(8,4*nrow+4), sharex=True)
            axes = [axes] if nrow==1 else axes
            axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\nhistogram [%]', fontsize=self.fig_property.fontsize)
            self.figs['histogram'] = Struct(fig=fig, axes=axes)
        
        if self.fig_property.plot_c:
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
        
        axes[1].plot(self.signal.t,self.filtered_signal.L2_error[m], color='magenta')
        axes[1].tick_params(labelsize=self.fig_property.fontsize)
        axes[1].set_ylim(find_limits(self.filtered_signal.L2_error[m]))
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
        sampling_number = np.random.choice(Np, size=int(Np*self.fig_property.particles_ratio), replace=False)

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

        def plot_histogram_Xn(self, ax, n=0, k=-1):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[:,n,:], n)
            ax.hist(state_of_filtered_signal[:,k], bins=self.fig_property.n_bins, color='blue', orientation='horizontal')
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X.shape[0], symbol=''))
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, n=n))
            return ax

        for n, ax in enumerate(axes):
            plot_histogram_Xn(self, ax, n)
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
    fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_histogram=True, plot_c=True)
    figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
    figs = figure.plot()

    plt.show()

