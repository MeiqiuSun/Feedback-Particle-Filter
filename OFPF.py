"""
Created on Thu Jun. 20, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division

import sympy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from FPF import FPF, Model, Galerkin
from Signal import Signal, Sinusoidal, LTI
from Tool import Struct, isiterable, find_limits

class OscillatedModel(Model):
    def __init__(self, frequency_range, amp_range, sigma_freq, sigma_amp, sigma_W):
        d = 2*len(amp_range)+1
        states = ['theta'] * d
        states_label = [r'$\theta$'] * d
        
        self.freq_min = frequency_range[0]
        self.freq_max = frequency_range[1]
        # X0_range = [[0,2*np.pi]] * d
        X0_range = [[0,0]] * d
        sigma_V = [sigma_freq] * d
        
        for m in range(len(amp_range)):
            states[2*m+1] = 'a{}'.format(m+1)
            states[2*m+2] = 'b{}'.format(m+1)
            states_label[2*m+1] = r'$a_{}$'.format(m+1)
            states_label[2*m+2] = r'$b_{}$'.format(m+1)
            X0_range[2*m+1] = amp_range[m]
            X0_range[2*m+2] = amp_range[m]
            sigma_V[2*m+1] = sigma_amp[m]
            sigma_V[2*m+2] = sigma_amp[m]
        
        Model.__init__(self, m=len(amp_range), d=d, X0_range=X0_range, 
                        states=states, states_label=states_label, 
                        sigma_V=sigma_V, sigma_W=sigma_W)
    
    def states_constraints(self, X):
        X[:,0] = np.mod(X[:,0], 2*np.pi)
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        dXdt[:,0] = np.linspace(self.freq_min*2*np.pi, self.freq_max*2*np.pi, X.shape[0])
        return dXdt

    def h(self, X):
        Y = np.einsum('im,i->im', X[:,1::2], np.cos(X[:,0])) + np.einsum('im,i->im', X[:,2::2], np.sin(X[:,0]))
        # Y = np.einsum('im,i->im', np.square(X[:,1::2]), np.cos(X[:,0])) + np.einsum('im,i->im', np.square(X[:,2::2]), np.sin(X[:,0]))
        return Y

class OscillatedGalerkin(Galerkin):
    """Galerkin approximation in finite element method:
            states: [ theta, a1, b1, a2, b2, ..., am, bm]
            base functions: [ a1*cos(theta), b1*sin(theta), ..., am*cos(theta), bm*sin(theta)] 
        Notation Note:
            Np: int, number of particles
            X: numpy array with the shape of (Np,d), particle states
        Initialize = Galerkin(states, m)
            states: 1D list of string with length d, name of each state
            m: int, number of observations
        Members = 
            d: int, number of states
            m: int, number of observations
            L: int, number of base functions
            states: 1D list of string with length d, name of each state
            result: temporary memory location of the result of exec()
            psi_list: 1D list of string with length L, sympy expression of each base function
            grad_psi_list: 2D list of string with length L-by-d, sympy expression of gradient of psi function
            grad_grad_psi_list: 3D list of string with length L-by-d-by-d, sympy expression of gradient of gradient of psi function
            psi_string: string, executable 1D list of numpy array with length L
            grad_psi_string, string, executable 2D list of numpy array with length L-by-d
            grad_grad_psi_string, string, executable 3D list of numpy array with length L-by-d-by-d
        Methods = 
            sympy_executable_result(): return sympy executable string
            split(X): split states (X) to members
            psi(X): return a numpy array with the shape of (L,Np), base functions
            grad_psi(X): return a numpy array with the shape of (L,d,Np), gradient of base functions
            grad_grad_psi(X): return a numpy array with the shape of (L,d,d,Np), gradient of gradient of base functions
    """
    def __init__(self, states, m):
        Galerkin.__init__(self, states, m, 2*m)
        
        # self.states = ['self.'+state for state in states]
        self.result = ''

        self.psi_list = [None for l in range(self.L)]
        self.grad_psi_list = [[None for d in range(self.d)] for l in range(self.L)]
        self.grad_grad_psi_list = [[[None for d2 in range(self.d)] for d1 in range(self.d)] for l in range(self.L)]  

        # initialize psi
        self.psi_string = '['
        for m in range(self.m):
            self.psi_list[2*m] = 'self.a{}*sympy.cos(self.theta)'.format(m+1)
            self.psi_list[2*m+1] = 'self.b{}*sympy.sin(self.theta)'.format(m+1)
            # self.psi_list[2*m] = 'self.a{}**2*sympy.cos(self.theta)'.format(m+1)
            # self.psi_list[2*m+1] = 'self.b{}**2*sympy.sin(self.theta)'.format(m+1)
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
            for d1 in range(self.d):
                exec('self.result = sympy.diff('+self.psi_list[l]+','+self.states[d1]+')')
                self.grad_psi_list[l][d1] = self.sympy_executable_result()
                
                self.grad_grad_psi_string += '['
                for d2 in range(self.d):
                    exec('self.result = sympy.diff('+self.grad_psi_list[l][d1]+','+self.states[d2]+')')
                    self.grad_grad_psi_list[l][d1][d2] = self.sympy_executable_result()
                self.grad_grad_psi_string += ((', ').join(self.grad_grad_psi_list[l][d1])).replace('sympy','np')
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

    def psi(self, X):
        self.split(X)
        exec("self.result = "+self.psi_string)
        trial_functions = self.result
        return np.array(trial_functions)
        
    def grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_psi_string)
        grad_trial_functions = np.zeros([self.L, self.d, X.shape[0]])
        for l in range(self.L):
            for d in range(self.d):
                grad_trial_functions[l,d,:] = self.result[l][d]
        return grad_trial_functions

    def grad_grad_psi(self, X):
        self.split(X)
        exec("self.result = "+self.grad_grad_psi_string)
        grad_grad_trial_functions = np.zeros([self.L, self.d, self.d, X.shape[0]])
        for l in range(self.L):
            for d1 in range(self.d):
                for d2 in range(self.d):
                    grad_grad_trial_functions[l,d1,d2,:] = self.result[l][d1][d2]
        return grad_grad_trial_functions

class OFPF(FPF):
    def __init__(self, number_of_particles, frequency_range, amp_range, sigma_amp, sigma_W, dt, sigma_freq=0):
        assert len(amp_range)==len(sigma_amp)==len(sigma_W),  \
            "length of amp_range ({}), length of sigma_amp ({}) and length of sigma_W ({}) do not match (should all be m)".format(
                len(amp_range), len(sigma_amp), len(sigma_W))
        model = OscillatedModel(frequency_range, amp_range, sigma_freq=sigma_freq, sigma_amp=sigma_amp, sigma_W=sigma_W)
        galerkin = OscillatedGalerkin(model.states, model.m)
        FPF.__init__(self, number_of_particles, model, galerkin, dt)
    
    def get_gain(self):
        gain = np.zeros(self.m, dtype=complex)
        for m in range(self.m):
            # gain[m] = complex(am,-bm)
            gain[m] = complex(np.mean(self.particles.X[:,2*m+1]),-np.mean(self.particles.X[:,2*m+2]))
            # gain[m] = complex(np.mean(np.square(self.particles.X[:,2*m+1])),-np.mean(np.square(self.particles.X[:,2*m+2])))
        return gain

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
        if 'particles_ratio' in self.fig_property.__dict__ :
            sampling_number = np.random.choice(Np, size=int(Np*self.fig_property.particles_ratio), replace=False)
        else :
            sampling_number = np.array(range(Np))

        def plot_Xn(self, ax, sampling_number, n=0):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[:,n,:], n)
            for i in range(sampling_number.shape[0]):
                if n==0:
                    ax.scatter(self.signal.t, state_of_filtered_signal[sampling_number[i]], s=0.5)
                else:
                    ax.scatter(self.signal.t, state_of_filtered_signal[sampling_number[i]], s=0.5)
                    # ax.scatter(self.signal.t, np.square(state_of_filtered_signal[sampling_number[i]]), s=0.5)
            ax.tick_params(labelsize=self.fig_property.fontsize)
            ax.set_ylim(self.set_limits(state_of_filtered_signal, n=n))
            return ax

        for n, ax in enumerate(axes):
            plot_Xn(self, ax, sampling_number, n)
        return axes

    def plot_histogram(self, axes):

        def plot_histogram_Xn(self, ax, bins, n=0, k=-1):
            state_of_filtered_signal = self.modified(self.filtered_signal.X[:,n,:], n)
            if k==0:
                ax.hist(state_of_filtered_signal[:,0], bins=bins, color='blue', orientation='horizontal')
            else:
                ax.hist(state_of_filtered_signal[:,k], bins=bins, color='blue', orientation='horizontal')
                # ax.hist(np.square(state_of_filtered_signal[:,k]), bins=bins, color='blue', orientation='horizontal')
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

def complex_to__mag_phase(gain):
    mag = np.abs(gain)
    phase = np.angle(gain)
    while phase[1]-phase[0] > 0:
        phase[0] += 2*np.pi
    return mag[1]/mag[0], phase[1]-phase[0]

def bode(omegas):
    mag = np.zeros(omegas.shape[0])
    phase = np.zeros(omegas.shape[0])
    X0=[[0.]]
    for i, omega in enumerate(omegas):
        dt = np.min((0.01/omega,0.02))
        sys = LTI(A=[[-1]], B=[[1]], C=[[1]], D=[[0]], sigma_V=[0.01], sigma_W=[0.01], dt=dt)
        T = 15/omega
        print("Frequency = {} [rad/s], sampling time = {} [s], Total time = {} [s], Total steps = {}".format(omega, dt, T, int(T/dt)))
        t = np.arange(0, T+dt, dt)
        f = omega/(2*np.pi)
        U = np.reshape(np.cos(2.*np.pi*f*t), (-1,t.shape[0]))
        Y, _ = sys.run(X0=X0, U=U)
        signal = Signal.create(t, U, Y)
        ofpf = OFPF(number_of_particles=1000, frequency_range=[f*0.9, f*1.1], amp_range=[[0.7,1.5],[0,1]],\
                    sigma_amp=[0.1, 0.1], sigma_W=[0.1, 0.1], dt=dt, sigma_freq=0)
        filtered_signal = ofpf.run(signal.Y, show_time=False)
        mag[i], phase[i] = complex_to__mag_phase(ofpf.get_gain())
        print(mag[i], phase[i])

        # fontsize = 20
        # fig_property = Struct(fontsize=fontsize, plot_signal=True, plot_X=True, plot_histogram=True, particles_ratio=0.01, plot_c=True)
        # figure = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal)
        # figs = figure.plot()
        # plt.show()
        
    return mag, phase

if __name__ == "__main__":
    # T = 200
    # t = np.arange(0, T+dt, dt)
    # f = 0.2/(2*np.pi)
    # U = np.reshape(np.cos(2.*np.pi*f*t), (-1,t.shape[0]))
    # X0=[[0.]]
    
    omegas = np.logspace(-1, 1, 200)
    # omegas = np.array([0.22])
    mag, phase = bode(omegas)
    np.savetxt('bode.txt', np.transpose(np.array([omegas, mag, phase])), fmt='%.4f')
    plt.figure()
    plt.semilogx(omegas, mag)
    plt.ylabel('mag')
    plt.xlabel('freq [rad/s]')
    plt.figure()
    plt.semilogx(omegas, phase)
    plt.ylabel('phase [rad]')
    plt.xlabel('freq [rad/s]')
    plt.show()
    # sys = LTI(A=[[-1]], B=[[1]], C=[[1]], D=[[0]], sigma_V=[0.], sigma_W=[0.], dt=dt)
    # Y, _ = sys.run(X0=X0, U=U)
    # signal = Signal.create(t, U, Y)


    # ofpf = OFPF(number_of_particles=1000, frequency_range=[f, f], amp_range=[[0.7,1.5],[0,1]],\
                # sigma_amp=[0.1, 0.1], sigma_W=[0.1, 0.1], dt=dt, sigma_freq=0)
    # filtered_signal = ofpf.run(signal.Y)

    # print(complex_to__mag_phase(ofpf.get_gain()))


    
    # plt.figure()
    # plt.plot(t, signal.Y[0,:], label=r'$U$')
    # plt.plot(t, filtered_signal.h_hat[0,:], label=r'$\hat U$')
    # plt.legend()

    # plt.figure()
    # plt.plot(t, signal.Y[1,:], label=r'$Y$')
    # plt.plot(t, filtered_signal.h_hat[1,:], label=r'$\hat Y$')
    # plt.legend()

    # plt.show()

    # print(galerkin.__dict__.keys())