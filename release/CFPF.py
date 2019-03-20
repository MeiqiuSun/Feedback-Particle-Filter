"""
Created on Wed Feb. 13, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct, isiterable, find_limits
from FPF import FPF

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

class CFPF(object):
    """CFPF: Coupled Feedback Partilce Filter
        Notation Note:
            M: number of observations(y)
        Initialize = CFPF(number_of_channels, number_of_particles, f_min, f_max, sigma_W, dt, h)
            number_of_channels: int, number of FPF channels
            number_of_particles: 1D list of int with length N, a list of number of particles for each channel
            f_min: 1D list of float with length N, a list of minimum frequencies in Hz for each channel
            f_max: 1D list of float with length N, a list of maximum frequencies in Hz for each channel
            sigma_W: 1D list of float with legnth M, standard deviations of noise of observations(y)
            dt: float, size of time step in sec
            h: function, estimated observation dynamics
        Members = 
            N: number of channels
            fpf: 1D list of FPF with length N, a list of N channels of FPF
            h_hat: numpy array with the shape of (M,), filtered observations
        Methods =
            calculate_h_hat(): Summation of h_hat for all FPF channels
            update(y): Update each FPF channel with new observation data y
            run(y): Run CFPF with time series observations(y)
    """

    def __init__(self, number_of_channels, numbers_of_particles, models, galerkins, sigma_B, sigma_W, dt):
        self.fpf = []
        self.Nc = number_of_channels
        min_sigma_W = np.max(np.array([models[j].min_sigma_W for j in range(self.Nc)]))
        self.sigma_W = np.array(sigma_W).clip(min=min_sigma_W)
        self.dt = dt
        for j in range(self.Nc):
            self.fpf.append(FPF(number_of_particles=numbers_of_particles[j], model=models[j], galerkin=galerkins[j], sigma_B=sigma_B[j], sigma_W=self.sigma_W, dt=self.dt))
        self.h_hat = np.zeros(len(self.sigma_W))
        return

    def calculate_h_hat(self):
        """Summation of h_hat for all FPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for j in range(self.Nc):        
            h_hat += self.fpf[j].h_hat
        return h_hat
    
    def update(self, y):
        """Update each FPF channel with new observation data y"""
        for j in range(self.Nc):
            self.fpf[j].update(y-self.h_hat+self.fpf[j].h_hat)
        self.h_hat = self.calculate_h_hat()
        return
    
    def run(self, y):
        """Run CFPF with time series observations(y)"""
        h_hat = np.zeros([self.Nc+1, y.shape[0], y.shape[1]])
        L2_error = np.zeros(y.shape)
        h = []
        X = []

        for j in range(self.Nc):
            h.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].M, y.shape[1]]))
            X.append(np.zeros([self.fpf[j].particles.Np, self.fpf[j].N, y.shape[1]]))
        
        for k in range(y.shape[1]):
            if np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(y[:,k])
            h_hat[0,:,k] = self.h_hat
            for j in range(self.Nc):
                h_hat[j+1,:,k] = self.fpf[j].h_hat
                h[j][:,:,k] = self.fpf[j].particles.h
                X[j][:,:,k] = self.fpf[j].particles.X
        
        for m in range(y.shape[0]):
            L2_error[m,:] = np.sqrt(np.cumsum(np.square(y[m,:]-h_hat[0,m,:])*self.dt)/(self.dt*(y.shape[1]-1)))/self.sigma_W[m]

        return Struct(h_hat=h_hat, L2_error=L2_error, h=h, X=X)

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
        
        axes[1].plot(self.signal.t,self.filtered_signal.L2_error[m], color='magenta')
        axes[1].tick_params(labelsize=self.fig_property.fontsize)
        axes[1].set_ylim(find_limits(self.filtered_signal.L2_error[m]))
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

if __name__ == "__main__":
    pass
    print('Nothing to show.')