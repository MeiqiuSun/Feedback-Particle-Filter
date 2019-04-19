"""
Created on Wed Mar. 5, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF
from Signal import Signal, Sinusoidals

import sympy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class COFPF(object):
    """COFPF: Coupled Oscillating Feedback Particle Filter
        Notation Note:
            N: int, number of states(X)
            M: int, number of observations(Y)
        Initialize = COFPF(number_of_channels, number_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt)
            number_of_channels: int, number of FPF channels
            number_of_particles: 1D list of int with length Nc, a list of number of particles for each channel
            amp_ranges: 2D list of float with length M-by-2, a list of amplitude range [amp_min, amp_max] for each observation (Ym)
            freq_ranges: 2D list of float with length Nc-by-2, a list of frequency range [f_min, f_max] for each channel
            sigma_B: 1D list of float with legnth N, standard deviations of process noise (same for each channel)
            sigma_W: 1D list of float with legnth M, standard deviations of noise of observations(Y)
            dt: float, size of time step in sec
            min_sigma_W: int, smallest value of sigma_W
        Members = 
            Nc: number of channels
            fpf: 1D list of FPF with length Nc, a list of FPF for each channel
            dt: float, size of time step in sec
            h_hat: numpy array with the shape of (M,), filtered observations
        Methods =
            calculate_h_hat(): Summation of h_hat for all FPF channels
            update(Y): Update each FPF channel with new observation data Y
            run(Y): Run COFPF with time series observations(Y)
    """

    def __init__(self, number_of_channels, numbers_of_particles, amp_ranges, freq_ranges, sigma_B, sigma_W, dt, min_sigma_W = 0.1):
        assert len(amp_ranges)==len(sigma_W), "length of amp_ranges ({}) and length of sigma_W ({}) do not match (should both be M)".format(len(amp_ranges), len(sigma_W))
        assert number_of_channels==len(freq_ranges), "number_of_channels ({}) and length of freq_ranges ({}) do not match (should both be Nc)".format(number_of_channels, len(freq_ranges))
        
        self.Nc = number_of_channels
        self.fpf = [None]*self.Nc
        
        self.dt = dt
        for j in range(self.Nc):
            model = self.Model(amp_ranges=amp_ranges, freq_range=freq_ranges[j])
            assert model.N==len(sigma_B), "model.N ({}) and length of sigma_B ({}) do not match (should both be N)".format(model.N, len(sigma_B))
            galerkin = self.Galerkin(states=model.states, M=model.M)
            self.fpf[j] = FPF(number_of_particles=numbers_of_particles[j], model=model, galerkin=galerkin, \
                              sigma_B=sigma_B, sigma_W=np.array(sigma_W).clip(min=min_sigma_W), dt=self.dt)
        self.h_hat = np.zeros(len(sigma_W))
        return
    
    class Model(object):
        """Model: states X^{(i)}=[ theta, r1, r2, ..., rM]^{(i)}
            Notation Note =
                Np: int, number of particles
                X: numpy array with the shape of (Np,N), particle states
            Initialize = Model(amp_ranges, freq_range, min_sigma_B=0.01, min_sigma_W=0.1)
                amp_range: 2D list of floats with length M-by-2, [min_amp, max_amp] for amplitude initialization of r1, ..., rM states
                freq_range: 1D list of floats with length 2, [min_freq, max_freq] for frequency initialization of state theta
                min_sigma_B: float, minimum standard deviation of process noise
                min_sigma_W: float, minimum standard deviation of observation noise
            Members = 
                M: int, number of observations
                N: int, number of states
                states: 1D list of string with lenght N, name of states
                states_label: 1D list of string with lenght N, LaTex label of states
                X0_range: 2D list of floats with length (M+1)-by-2, [ [0,2pi], [min_r1,max_r1], [min_r2,max_r2], ..., [min_rM,max_rM]]
                freq_range: 1D list of floats with length 2, [min_freq, max_freq] for frequency initialization of state theta
                min_sigma_B: float, minimum standard deviation of process noise
                min_sigma_W: float, minimum standard deviation of observation noise
            Methods =
                modified_sigma_B(sigma_B): return a numpy array with the shape of (N,), set process noise with minimum value min_sigma_B
                modified_sigma_W(sigma_W): return a numpy array with the shape of (M,), set observation noise with minimum value value min_sigma_W
                states_constraints(X): return a numpy array with the shape of (Np,N), constrainted states
                f(X, t): return a numpy array with the shape of (Np,N), theta^{(i)}_dot = omega^{(i)} for i^th particle
                h(X): return a numpy array with the shape of (Np,M), [ r1*cos(theta), r2*cos(theta), ..., rM*cos(theta)]
        """
        def __init__(self, amp_ranges, freq_range, min_sigma_B=0.01, min_sigma_W=0.1):
            self.M = len(amp_ranges)
            self.N = self.M+1

            self.states = ['r{}'.format(m+1) for m in range(self.M)]
            self.states.insert(0, 'theta')
            self.states_label = [r'$r_{}$'.format(m+1) for m in range(self.M)]
            self.states_label.insert(0, r'$\theta$')
            
            self.X0_range = [amp_ranges[m] for m in range(self.M)]
            self.X0_range.insert(0,[0,2*np.pi])
            self.freq_range = freq_range
            
            self.min_sigma_B = min_sigma_B
            self.min_sigma_W = min_sigma_W

        def modified_sigma_B(self, sigma_B):
            sigma_B = sigma_B.clip(min=self.min_sigma_B)
            return sigma_B
        
        def modified_sigma_W(self, sigma_W):
            sigma_W = sigma_W.clip(min=self.min_sigma_W)
            return sigma_W

        def states_constraints(self, X):
            X[:,0] = np.mod(X[:,0], 2*np.pi)
            return X

        def f(self, X):
            dXdt = np.zeros(X.shape)
            dXdt[:,0] = np.linspace(2*np.pi*self.freq_range[0], 2*np.pi*self.freq_range[1], X.shape[0])
            return dXdt

        def h(self, X):
            Y = np.zeros([X.shape[0], self.M])
            for m in range(self.M):
                Y[:,m] = (X[:,m+1])**2*np.cos(X[:,0]) 
            return Y
    
    class Galerkin(object):
        """Galerkin approximation in finite element method:
                states: [ theta, r1, r2, ..., rM]
                base functions: [ rm*cos(theta), rm*sin(theta)] with m = 1, 2, ..., M
            Notation Note:
                Np: int, number of particles
                X: numpy array with the shape of (Np,N), particle states
            Initialize = Galerkin(N,M)
                states: 1D list of string with length N, name of each state
                M: int, number of observations
            Members = 
                N: int, number of states
                M: int, number of observations
                L: int, number of base functions
                states: 1D list of string with length N, name of each state
                result: temporary memory location of the result of exec()
                psi_list: 1D list of string with length L, sympy expression of each base function
                grad_psi_list: 2D list of string with length L-by-N, sympy expression of gradient of psi function
                grad_grad_psi_list: 3D list of string with length L-by-N-by-N, sympy expression of gradient of gradient of psi function
                psi_string: string, executable 1D list of numpy array with length L
                grad_psi_string, string, executable 2D list of numpy array with length L-by-N
                grad_grad_psi_string, string, executable 3D list of numpy array with length L-by-N-by-N
            Methods = 
                sympy_executable_result(): return sympy executable string
                split(X): split states (X) to inherited members (self.states[n], n = 1, 2, ..., N)
                psi(X): return a numpy array with the shape of (L,Np), base functions
                grad_psi(X): return a numpy array with the shape of (L,N,Np), gradient of base functions
                grad_grad_psi(X): return a numpy array with the shape of (L,N,N,Np), gradient of gradient of base functions
        """
        def __init__(self, states, M):
            self.N = len(states)
            self.M = M
            self.L = 2*self.M
            
            self.states = ['self.'+state for state in states]
            self.result = ''

            self.psi_list = [None for l in range(self.L)]
            self.grad_psi_list = [[None for n in range(self.N)] for l in range(self.L)]
            self.grad_grad_psi_list = [[[None for n2 in range(self.N)] for n1 in range(self.N)] for l in range(self.L)]  

            # initialize psi
            self.psi_string = '['
            for m in range(self.M):
                self.psi_list[2*m] = 'self.r{}**2*sympy.cos(self.theta)'.format(m+1)
                self.psi_list[2*m+1] = 'self.r{}**2*sympy.sin(self.theta)'.format(m+1)
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
                for n1 in range(self.N):
                    exec('self.result = sympy.diff('+self.psi_list[l]+','+self.states[n1]+')')
                    self.grad_psi_list[l][n1] = self.sympy_executable_result()
                    
                    self.grad_grad_psi_string += '['
                    for n2 in range(self.N):
                        exec('self.result = sympy.diff('+self.grad_psi_list[l][n1]+','+self.states[n2]+')')
                        self.grad_grad_psi_list[l][n1][n2] = self.sympy_executable_result()
                    self.grad_grad_psi_string += ((', ').join(self.grad_grad_psi_list[l][n1])).replace('sympy','np')
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

        def split(self, X):
            for n in range(self.N):
                exec(self.states[n]+' = X[:,{}]'.format(n))

        def psi(self, X):
            self.split(X)
            exec("self.result = "+self.psi_string)
            trial_functions = self.result
            return np.array(trial_functions)
            
        def grad_psi(self, X):
            self.split(X)
            exec("self.result = "+self.grad_psi_string)
            grad_trial_functions = np.zeros([self.L, self.N, X.shape[0]])
            for l in range(self.L):
                for n in range(self.N):
                    grad_trial_functions[l,n,:] = self.result[l][n]
            return grad_trial_functions

        def grad_grad_psi(self, X):
            self.split(X)
            exec("self.result = "+self.grad_grad_psi_string)
            grad_grad_trial_functions = np.zeros([self.L, self.N, self.N, X.shape[0]])
            for l in range(self.L):
                for n1 in range(self.N):
                    for n2 in range(self.N):
                        grad_grad_trial_functions[l,n1,n2,:] = self.result[l][n1][n2]
            return grad_grad_trial_functions

    def calculate_h_hat(self):
        """Summation of h_hat for all FPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for j in range(self.Nc):        
            h_hat += self.fpf[j].h_hat
        return h_hat
    
    def update(self, Y):
        """Update each FPF channel with new observation data y"""
        for j in range(self.Nc):
            self.fpf[j].update(Y-self.h_hat+self.fpf[j].h_hat)
        self.h_hat = self.calculate_h_hat()
        return
    
    def run(self, Y):
        """Run COFPF with time series observations(Y)"""
        h_hat = np.zeros([self.Nc+1, Y.shape[0], Y.shape[1]])
        error = np.zeros(Y.shape)
        h = []
        X = []

        for j in range(self.Nc):
            h.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].M, Y.shape[1]]))
            X.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].N, Y.shape[1]]))
        
        for k in range(Y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(Y[:,k])
            h_hat[0,:,k] = self.h_hat
            for j in range(self.Nc):
                h_hat[j+1,:,k] = self.fpf[j].h_hat
                h[j][:,:,k] = self.fpf[j].particles.h
                X[j][:,:,k] = self.fpf[j].particles.X
        
        for m in range(Y.shape[0]):
            error[m,:] = Y[m,:]-h_hat[0,m,:]

        return Struct(h_hat=h_hat, error=error, h=h, X=X)

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
        print("plot_signal")
        if self.fig_property.plot_signal:
            for m in range(self.signal.Y.shape[0]):
                fig = plt.figure(figsize=(8,8))
                ax3 = plt.subplot2grid((5,1),(4,0))
                ax2 = plt.subplot2grid((5,1),(3,0), sharex=ax3)
                ax1 = plt.subplot2grid((5,1),(0,0),rowspan=3, sharex=ax3)
                axes = self.plot_signal([ax1, ax2, ax3], m)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
                fig.add_subplot(111, frameon=False)
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
                plt.title('m={}'.format(m+1), fontsize=self.fig_property.fontsize+2)
                self.figs['signal_{}'.format(m+1)] = Struct(fig=fig, axes=axes)

        print("plot_X")
        if self.fig_property.plot_X:
            ncol = len(self.filtered_signal.X)
            maxrow = np.max(np.array([self.filtered_signal.X[j].shape[1] for j in range(ncol)]))
            fig = plt.figure(figsize=(8*ncol, 4*maxrow+4))
            axes = []
            for j in range(ncol):
                nrow = self.filtered_signal.X[j].shape[1]
                axes.append([])
                for n in range(nrow):
                    if n==0:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j)))
                    else:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j), sharex=axes[j][0]))
                    if not n==(nrow-1):
                        plt.setp(axes[j][n].get_xticklabels(), visible=False)
                axes[j][0].set_title('channel {}'.format(j+1), fontsize=self.fig_property.fontsize)

            axes = self.plot_X(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\ntime [s]', fontsize=self.fig_property.fontsize)
            self.figs['X'] = Struct(fig=fig, axes=axes)

        print("plot_histogram")
        if self.fig_property.plot_histogram:
            ncol = len(self.filtered_signal.X)
            maxrow = np.max(np.array([self.filtered_signal.X[j].shape[1] for j in range(ncol)]))
            fig = plt.figure(figsize=(8*ncol, 4*maxrow+4))
            axes = []
            for j in range(ncol):
                nrow = self.filtered_signal.X[j].shape[1]
                axes.append([])
                for n in range(nrow):
                    if n==0:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j)))
                    else:
                        axes[j].append(plt.subplot2grid((nrow,ncol),(n,j), sharex=axes[j][0]))
                    if not n==(nrow-1):
                        plt.setp(axes[j][n].get_xticklabels(), visible=False)
                axes[j][0].set_title('channel {}'.format(j+1), fontsize=self.fig_property.fontsize)
                        
            axes = self.plot_histogram(axes)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('\nhistogram [%]', fontsize=self.fig_property.fontsize)
            self.figs['histogram'] = Struct(fig=fig, axes=axes)
            
        return self.figs

    def plot_signal(self, axes, m):
        axes[0].plot(self.signal.t, self.signal.Y[m], color='gray', label=r'signal $Y_t$')
        axes[0].plot(self.signal.t, self.filtered_signal.h_hat[0,m], color='magenta', label=r'estimation $\hat h_t$')
        axes[0].legend(fontsize=self.fig_property.fontsize-5, loc='lower right', ncol=2)
        axes[0].tick_params(labelsize=self.fig_property.fontsize)
        axes[0].set_ylim(find_limits(self.signal.Y[m]))
        
        axes[1].plot(self.signal.t,self.filtered_signal.error[m], color='magenta')
        axes[1].tick_params(labelsize=self.fig_property.fontsize)
        axes[1].set_ylim(find_limits(self.filtered_signal.error[m]))
        axes[1].set_ylabel('$L_2$ error', fontsize=self.fig_property.fontsize)

        # for j in range(self.filtered_signal.h_hat.shape[0]-1):
        #     for i in range(self.filtered_signal.h[j].shape[0]):
        #         axes[2].scatter(self.signal.t, self.filtered_signal.h[j][i,m,:], color='C{}'.format(j+1), alpha=0.01)
        for j in range(self.filtered_signal.h_hat.shape[0]-1):
            axes[2].plot(self.signal.t, self.filtered_signal.h_hat[j+1,m,:], color='C{}'.format(j), label=r'$\hat h_{}$'.format(j+1))
        axes[2].tick_params(labelsize=self.fig_property.fontsize)
        axes[2].set_ylim(find_limits(self.filtered_signal.h_hat[1:,m]))
        axes[2].legend(fontsize=self.fig_property.fontsize-5, loc='center right')
        return axes

    def modified(self, signal, j, n):
        if not 'restrictions' in self.fig_property.keys():
            return signal
        else:
            signal = self.fig_property.restrictions.scaling[j][n]*signal
            if not np.isnan(self.fig_property.restrictions.mod[j][n]):
                signal = np.mod(signal, self.fig_property.restrictions.mod[j][n])
        return signal

    def set_limits(self, signal, j, n=np.nan, center=False):
        limits = find_limits(signal)
        if not 'restrictions' in self.fig_property.keys():
            return limits
        if not np.isnan(n):
            if not np.isnan(self.fig_property.restrictions.mod[j][n]):
                limits = find_limits(np.array([0, self.fig_property.restrictions.mod[j][n]]))
            if self.fig_property.restrictions.zero[j][n]:
                limits = find_limits(np.append(signal,0))
        if center or (self.fig_property.restrictions.center[j][n] if not np.isnan(n) else False):
            maximum = np.max(np.absolute(limits))
            limits = [-maximum, maximum]
        return limits

    def plot_X(self, axes):

        def plot_Xn(self, ax, sampling_number, j, n):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[j][:,n,:], j, n)
            for i in range(sampling_number.shape[0]):
                ax.scatter(self.signal.t, state_of_filtered_signal[sampling_number[i]], s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, j, n=n))
            return ax

        for j in range(len(axes)):
            Np = self.filtered_signal.X[j].shape[0]
            sampling_number = np.random.choice(Np, size=int(Np*self.fig_property.particles_ratio), replace=False)
            for n, ax in enumerate(axes[j]):
                plot_Xn(self, ax, sampling_number, j, n)
        return axes

    def plot_histogram(self, axes):

        def plot_histogram_Xn(self, ax, j=0, n=0, k=-1):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[j][:,n,:], j, n)
            ax.hist(state_of_filtered_signal[:,k], bins=self.fig_property.n_bins, color='blue', orientation='horizontal')
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=self.filtered_signal.X[j].shape[0], symbol=''))
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, j, n=n))
            return ax
        
        max_percentage = 0
        for j in range(len(axes)):
            for n, ax in enumerate(axes[j]):
                plot_histogram_Xn(self, ax, j, n)
                max_percentage = np.max([max_percentage, ax.get_xlim()[1]])
        for j in range(len(axes)):
            for ax in axes[j]:
                ax.set_xlim([0, max_percentage])

        return axes
def restrictions(states):
    mod = []
    scaling = []
    center = []
    zero = []
    for j in range(len(states)):
        mod.append([])
        scaling.append([])
        center.append([])
        zero.append([])
        for state in states[j]:
            mod[j].append(2*np.pi if 'theta' in state else np.nan)
            scaling[j].append(1/(2*np.pi) if 'omega' in state else 1)
            center[j].append('r' in state)
            zero[j].append('omega' in state)
    return Struct(mod=mod, scaling=scaling, center=center, zero=zero)

def set_yaxis(figs, models):
    for j in range(len(models)):
        for n, state in enumerate(models[j].states):
            if 'X' in figs.keys():
                figs['X'].axes[j][n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
            if 'histogram' in figs.keys():
                figs['histogram'].axes[j][n].set_ylabel(state, fontsize=figs['fontsize'], rotation=0)
            if 'theta' in state:
                if 'X' in figs.keys():
                    figs['X'].axes[j][n].set_yticks(np.linspace(0,2*np.pi,5))
                    figs['X'].axes[j][n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
                if 'histogram' in figs.keys():
                    figs['histogram'].axes[j][n].set_yticks(np.linspace(0,2*np.pi,5))
                    figs['histogram'].axes[j][n].set_yticklabels([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
            if 'omega' in state:
                pass
    return figs

    
if __name__ == "__main__":
    T = 50.
    fs = 160
    dt = 1/fs
    signal_type1 = Sinusoidals(dt, amp=[1,3], freq=[1.2,3.8])
    signal_type2 = Sinusoidals(dt, amp=[2], freq=[1.2])
    signal = Signal(signal_type1, T) + Signal(signal_type2, T)
    
    # cofpf = COFPF(number_of_channels=2, numbers_of_particles=[1000,1000], amp_ranges=[[0,4]], freq_ranges=[[1,2],[3,4]], sigma_B=[0, 1], sigma_W=[0.2], dt=signal.dt)
    cofpf = COFPF(number_of_channels=2, numbers_of_particles=[1000,1000], amp_ranges=[[0,2],[0,2]], freq_ranges=[[1,2],[3,4]], sigma_B=[0, 0.1, 0.1], sigma_W=[1,1], dt=signal.dt)
    filtered_signal = cofpf.run(signal.Y)
    
    fontsize = 20
    fig_property = Struct(fontsize=fontsize, show=False, plot_signal=True, plot_X=True, particles_ratio=0.01,\
                          plot_histogram=True, n_bins = 100, plot_c=False)
    figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
    figs['X'].axes[0][1].axhline(np.sqrt(signal_type1.amp[0]), color='black', linestyle='--')
    figs['X'].axes[0][1].plot(signal.t, np.mean(filtered_signal.X[0][:,1,:], axis=0), color='blue', linewidth=2)
    figs['X'].axes[0][2].axhline(np.sqrt(signal_type2.amp[0]), color='black', linestyle='--')
    figs['X'].axes[0][2].plot(signal.t, np.mean(filtered_signal.X[0][:,2,:], axis=0), color='blue', linewidth=2)
    figs['X'].axes[1][1].axhline(np.sqrt(signal_type1.amp[1]), color='black', linestyle='--')
    figs['X'].axes[1][1].plot(signal.t, np.mean(filtered_signal.X[1][:,1,:], axis=0), color='blue', linewidth=2)    
    figs['X'].axes[1][2].axhline(0, color='black', linestyle='--')
    figs['X'].axes[1][2].plot(signal.t, np.mean(filtered_signal.X[1][:,2,:], axis=0), color='blue', linewidth=2)
    figs['histogram'].axes[0][1].axhline(np.sqrt(signal_type1.amp[0]), color='black', linestyle='--')
    figs['histogram'].axes[0][2].axhline(np.sqrt(signal_type2.amp[0]), color='black', linestyle='--')
    figs['histogram'].axes[1][1].axhline(np.sqrt(signal_type1.amp[1]), color='black', linestyle='--')
    figs['histogram'].axes[1][2].axhline(0, color='black', linestyle='--')
    plt.show()
    # figs = set_yaxis(figs, cofpo.model)


    