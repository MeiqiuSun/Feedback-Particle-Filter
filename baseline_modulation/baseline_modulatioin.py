"""
Created on Mon Apr. 01, 2019
@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors

import pickle

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from Tool import Struct, find_limits
from modulations import Modulations, plot_signal
from Signal import Signal
from FPF import FPF, Figure

class Model(object):           # 1 state (c)
    def __init__(self, c_range, sigma_B=0.01, min_sigma_W=0.1):
        self.states = [r'$c$']
        self.N = len(self.states)
        self.M = 1
        self.X0_range = [c_range]
        self.sigma_B = [sigma_B]
        self.min_sigma_W = min_sigma_W

    def modified_sigma_B(self, sigma_B):
        return self.sigma_B
    
    def modified_sigma_W(self, sigma_W):
        sigma_W = sigma_W.clip(min=self.min_sigma_W)
        return sigma_W

    def states_constraints(self, X):
        return X

    def f(self, X):
        dXdt = np.zeros(X.shape)
        return dXdt

    def h(self, X):
        return X[:,0]

class Galerkin(object):       #  1 state (c) 1 base functions [c]
    # Galerkin approximation in finite element method
    def __init__(self):
        # L: int, number of trial (base) functions
        self.L = 1       # trial (base) functions
    
    def split(self, X):
        c = X[:,0]
        zeros = np.zeros(X.shape[0]) 
        ones = np.ones(X.shape[0])
        return c, zeros, ones
        
    def psi(self, X):
        c, _, _ = self.split(X)
        trial_functions = [c]
        return np.array(trial_functions)
        
    # gradient of trial (base) functions
    def grad_psi(self, X):
        c, zeros, ones = self.split(X)
        grad_trial_functions = [[ones]]
        return np.array(grad_trial_functions)

    # gradient of gradient of trail (base) functions
    def grad_grad_psi(self, X):
        c, zeros, ones = self.split(X)
        grad_grad_trial_functions = [[[zeros]]]
        return np.array(grad_grad_trial_functions)

def generate_signals(c0=0, amp_c=1, freq_c=1, sigma_B=0.1, n=10):
        
    T = 20.
    fs = 100
    dt = 1/fs
    rho = 10
    
    X0 = [0, 0, 0, c0]
    sigma_B = [0, 0, 0, sigma_B]

    signal_type = Modulations(dt, rho, X0, sigma_B, amp_r=0, freq_r=0, amp_omega=0, freq_omega=0, amp_c=amp_c, freq_c=freq_c)
    with open('signals/signal_type.pickle', 'wb') as handle:
        pickle.dump(signal_type, handle)
    
    print("Generating Signal(s) ...")
    for k in range(n):
        signal = Signal(signal_type=signal_type, T=T)
        with open('signals/signal{}.pickle'.format(k), 'wb') as handle:
            pickle.dump(signal, handle)
    print("Finish Generating Signal(s)!")
    return signal_type, signal

def filter_signals(n=5):
    signal_type = pickle.load(open('signals/signal_type.pickle', 'rb'))
    c0 = signal_type.X0[3]
    amp_c = signal_type.amp_c
    n_c = 15
    c_range = np.zeros([n_c, 2])
    for i in range(n_c):
        d_amp_c = 2*amp_c*i/(n_c-1)
        c_range[i,:] = np.array([c0-d_amp_c, c0+d_amp_c])
    n_sigma_B = 15
    sigma_B = np.logspace(-2, 1, num=n_sigma_B, endpoint=True)
    n_trial = 5
    with open('filtered_signals/parameters.pickle', 'wb') as handle:
        pickle.dump(Struct(c_range=c_range, sigma_B=sigma_B, n_trial=n_trial), handle)
    
    for k in range(n):
        with open('signals/signal{}.pickle'.format(k), 'rb') as handle:
            signal = pickle.load(handle)
            for i in range(n_sigma_B):
                for j in range(n_c):
                    folder = 'filtered_signals/signal{}/'.format(k)+'sigmaB{}c{}'.format(i,j)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    print('filtering '+folder)
                    model = Model(c_range=c_range[j,:], sigma_B=sigma_B[i])
                    for l in range(n_trial):
                        fpf = FPF(number_of_particles=1000, model=model, galerkin=Galerkin(), sigma_B=signal_type.sigma_B, sigma_W=signal_type.sigma_W, dt=signal_type.dt)
                        filtered_signal = fpf.run(signal.Y)
                        with open(folder+'/trial{}.pickle'.format(l), 'wb') as handle:
                            pickle.dump(Struct(model=model, fpf=fpf, filtered_signal=filtered_signal), handle)
    return

def process_signals():
    n = len(os.listdir('filtered_signals'))-1
    parameters = pickle.load(open('filtered_signals/parameters.pickle', 'rb'))
    sigma_B = parameters.sigma_B
    c_range = parameters.c_range
    n_trial = parameters.n_trial
    average_var = np.zeros([sigma_B.shape[0], c_range.shape[0], n, n_trial])
    mean_var = np.zeros([sigma_B.shape[0], c_range.shape[0]])
    for i in range(sigma_B.shape[0]):
        for j in range(c_range.shape[0]):
            print('processing sigma_B={} c_range=[{},{}]'.format(sigma_B[i], parameters.c_range[j,0], parameters.c_range[j,1]))
            for k in range(n):
                signal = pickle.load(open('signals/signal{}.pickle'.format(k), 'rb'))
                for l in range(n_trial):
                    data = pickle.load(open('filtered_signals/signal{}/sigmaB{}c{}/trial{}.pickle'.format(k,i,j,l), 'rb'))
                    filtered_signal = data.filtered_signal
                    average_var[i,j,k,l] = np.mean(np.var(filtered_signal.X[:,0,int(signal.t.shape[0]/2):], axis=0))
            mean_var[i,j] = np.mean(average_var[i,j,:,:])
    with open('processed_signals.pickle', 'wb') as handle:
        pickle.dump(Struct(sigma_B=sigma_B, c_range=c_range, average_var=average_var, mean_var=mean_var), handle)
    return

def plot_variance(fontsize = 20, signal_name = 'signal0', savefig=False):
    data = pickle.load(open('processed_signals.pickle', 'rb'))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    vmin, vmax = find_limits(np.array([data.mean_var.min(), data.mean_var.max()]), scale='log')

    pcm = ax.pcolormesh(data.sigma_B, data.c_range[:,1]-data.c_range[:,0], np.transpose(data.mean_var), \
                        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.get_cmap('Reds'))
    
    cbar = fig.colorbar(pcm, ax=ax, extend='max')
    cbar.ax.tick_params(labelsize=fontsize) 
    ax.set_xscale('log')
    # ax.set_xlim(find_limits(ax.get_xticks(), scale='log'))
    ax.set_ylim(find_limits(ax.get_yticks()))
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r'$\sigma_B^{(i)}$', fontsize=fontsize)
    ax.set_ylabel(r'$c_0^{(i)}$ range', fontsize=fontsize)
    ax.set_title(r'$Var(\hat h_t)$', fontsize=fontsize+2)
    if savefig:
        print('saving variance figure...')
        plt.savefig('variance.png', dpi=200)
    
    signal = pickle.load(open('signals/'+signal_name+'.pickle', 'rb'))
    fig_property = Struct(fontsize=fontsize, plot_signal=False, plot_X=True, particles_ratio=0.1, \
                          plot_histogram=False, n_bins = 100, plot_c=False)
    for i in range(3):
        for j in range(3):
            with open('filtered_signals/'+signal_name+'/sigmaB{}c{}/trial0.pickle'.format(int((data.sigma_B.shape[0]-1)*i/2), int((data.c_range.shape[0]-1)*j/2)), 'rb') as handle:
                filtered_signal = pickle.load(handle).filtered_signal
                figs = Figure(fig_property=fig_property, signal=signal, filtered_signal=filtered_signal).plot()
                figs['X'].axes[0].plot(signal.t, signal.Y[0], color='gray', label=r'signal $Y_t$')
                figs['X'].axes[0].plot(signal.t, filtered_signal.h_hat[0], color='red', label=r'estimation $\hat h_t$')
                figs['X'].axes[0].legend(fontsize=fontsize-5, loc='lower right', ncol=2)
                figs['X'].axes[0].tick_params(labelsize=fontsize)
                figs['X'].axes[0].set_ylim(find_limits(np.array([-3, 2])))
                figs['X'].axes[0].set_title(r'$\sigma_B={:.2f},\ \ c_0\sim$uniform$({:.2f},\ {:.2f})$'.format(data.sigma_B[int((data.sigma_B.shape[0]-1)*i/2)], \
                                                                                    data.c_range[int((data.c_range.shape[0]-1)*j/2),0], \
                                                                                    data.c_range[int((data.c_range.shape[0]-1)*j/2),1]), fontsize=fontsize+2)
                if savefig:
                    print('saving sigmaBn{}cn{} figure...'.format(i,j))
                    plt.savefig('sigmaBn{}cn{}.png'.format(i,j), dpi=200)
    return

def final_graph():
    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    for i in range(3):
        for j in range(3):
            if (i==1) and (j==1):
                img=mpimg.imread('variance.png')
            else:
                img=mpimg.imread('sigmaBn{}cn{}.png'.format(i,j))
            axes[i,j].imshow(img) 
    return

if __name__ == "__main__":

    # generate_signals()
    filter_signals()
    process_signals()
    plot_variance(savefig=True)
    # final_graph()
    plt.show()

