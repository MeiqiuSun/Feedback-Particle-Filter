"""
Created on Tue Jun. 24, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from Tool import Struct
from OFPF import OFPF, complex_to_mag_phase
from Signal import Signal, LTI

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter

class COFPF(object):
    """COFPF: Coupled Oscillating Feedback Particle Filter
        Initialize = COFPF(ofpfs, sigma_W, dt)
            ofpfs: 1D list of Struct, properties of OFPF
            sigma_W: 1D list of float with legnth m, standard deviations of noise of observations(Y)
            dt: float, size of time step in sec
        Members = 
            Nc: number of channels
            ofpfs: 1D list of OFPF with length Nc, a list of OFPF for each channel
            dt: float, size of time step in sec
            h_hat: numpy array with the shape of (m,), filtered observations
        Methods =
            calculate_h_hat(): Summation of h_hat for all OFPF channels
            update(Y): Update each OFPF channel with new observation data Y
            run(Y): Run COFPF with time series observations(Y)
    """

    def __init__(self, ofpfs, sigma_W, dt):
        self.Nc = len(ofpfs)
        self.ofpfs = [None]*self.Nc
        for j in range(self.Nc):
            self.ofpfs[j] = OFPF(number_of_particles=ofpfs[j].number_of_particles, 
                                    frequency_range=ofpfs[j].frequency_range, 
                                    amp_range=ofpfs[j].amp_range, 
                                    sigma_amp=ofpfs[j].sigma_amp, 
                                    sigma_W=sigma_W, dt=dt, sigma_freq=ofpfs[j].sigma_freq)
        self.h_hat = np.zeros(len(sigma_W))
        self.dt = dt
        return
    
    def calculate_h_hat(self):
        """Summation of h_hat for all OFPF channels"""
        h_hat = np.zeros(self.h_hat.shape)
        for j in range(self.Nc):        
            h_hat += self.ofpfs[j].h_hat
        return h_hat
    
    def update(self, Y):
        """Update each FPF channel with new observation data y"""
        for j in range(self.Nc):
            self.ofpfs[j].update(Y-self.h_hat+self.ofpfs[j].h_hat)
        self.h_hat = self.calculate_h_hat()
        return
    
    def run(self, Y, show_time=False):
        """Run COFPF with time series observations(Y)"""
        h_hat = np.zeros(Y.shape)
        ofpfs = [None] * self.Nc

        for j in range(self.Nc):
            ofpfs[j] = Struct(h_hat=np.zeros(Y.shape), 
                                h=np.zeros((self.ofpfs[j].particles.Np, self.ofpfs[j].m, Y.shape[1])),
                                X=np.zeros((self.ofpfs[j].particles.Np, self.ofpfs[j].d, Y.shape[1])))
        
        for k in range(Y.shape[1]):
            if show_time and np.mod(k,int(1/self.dt))==0:
                print("===========time: {} [s]==============".format(int(k*self.dt)))
            self.update(Y[:,k])
            h_hat[:,k] = self.h_hat
            for j in range(self.Nc):
                ofpfs[j].h_hat[:,k] = self.ofpfs[j].h_hat
                ofpfs[j].h[:,:,k] = self.ofpfs[j].particles.h
                ofpfs[j].X[:,:,k] = self.ofpfs[j].particles.X

        return Struct(h_hat=h_hat, ofpfs=ofpfs)

def bode(omegas):
    mag = np.zeros(omegas.shape)
    phase = np.zeros(omegas.shape)
    X0 = [[0.]]
    for i in range(omegas.shape[1]):
        dt = np.min((0.01/np.max(omegas[:,i]),0.01))

        sys = LTI(A=[[-1]], B=[[1]], C=[[1]], D=[[0]], sigma_V=[0.01], sigma_W=[0.01], dt=dt)
        T = 1*2*np.pi/np.min(omegas[:,i])
        t = np.arange(0, T+dt, dt)
        U = np.zeros(t.shape)
        for j in range(omegas[:,i].shape[0]):
            print("Frequency_{} = {} [rad/s], ".format(j+1,omegas[j,i]))
            U += np.cos(omegas[j,i]*t)
        U = np.reshape(U, (-1, t.shape[0]))
        print("sampling time = {} [s], Total time = {} [s], Total steps = {}".format(dt, T, int(T/dt)))
        Y, _ = sys.run(X0=X0, U=U)
        signal = Signal.create(t, U, Y)
        
        ofpfs = [None] * omegas[:,i].shape[0]
        for j in range(omegas[:,i].shape[0]):
            f = omegas[j,i]/(2*np.pi)
            ofpfs[j] = Struct(number_of_particles=100, frequency_range=[f,f], amp_range=[[0.9,1.1],[0.9/(omegas[j,i]**2+1),1.1*omegas[j,i]/(omegas[j,i]**2+1)]], sigma_amp=[0.1, 0.1], sigma_freq=0.1)
        cofpf = COFPF(ofpfs, sigma_W=[0.1, 0.1], dt=dt)
        filtered_signal = cofpf.run(signal.Y, show_time=False)
        for j in range(omegas[:,i].shape[0]):
            mag[j,i], phase[j,i] = complex_to_mag_phase(cofpf.ofpfs[j].get_gain())
            print(mag[j,i], phase[j,i])
    return mag, phase


if __name__ == "__main__":
    n=101
    omegas = np.zeros((3,n))
    omegas[0,:] = np.logspace(-1,0,n)
    omegas[1,:] = np.logspace(0,1,n)
    omegas[2,:] = np.logspace(1,2,n)
    mag, phase = bode(omegas)
    
    for j in range(omegas.shape[0]):
        np.savetxt('bode{}{}.txt'.format(omegas.shape[0],j+1), np.transpose(np.array([omegas[j,:], mag[j,:], phase[j,:]])), fmt='%.4f')
        plt.figure(1)
        plt.semilogx(omegas[j,:], mag[j,:])
        plt.ylabel('mag')
        plt.xlabel('freq [rad/s]')
        plt.figure(2)
        plt.semilogx(omegas[j,:], phase[j,:])
        plt.ylabel('phase [rad]')
        plt.xlabel('freq [rad/s]')
    plt.show()
    # X0=[[0.]]
    # dt = 0.01
    # T = 30
    # t = np.arange(0, T+dt, dt)
    # f = omega/(2*np.pi)
    # U = np.reshape(np.cos(2.*np.pi*f*t)+np.cos(2.*np.pi*(f*2)*t), (-1,t.shape[0]))
    # sys = LTI(A=[[-10]], B=[[10]], C=[[1]], D=[[0]], sigma_V=[0.01], sigma_W=[0.01], dt=dt)
    # Y, _ = sys.run(X0=X0, U=U)
    # signal = Signal.create(t, U, Y)
    # ofpfs = [None]*2
    # ofpfs[0] = Struct(number_of_particles=100, frequency_range=[f*0.9,f*1.1], amp_range=[[0.7,1.5],[0,1]], sigma_amp=[0.1, 0.1], sigma_freq=0)
    # ofpfs[1] = Struct(number_of_particles=100, frequency_range=[f*2*0.9,f*2*1.1], amp_range=[[0.7,1.5],[0,1]], sigma_amp=[0.1, 0.1], sigma_freq=0)
    # cofpf = COFPF(ofpfs, sigma_W=[0.1, 0.1], dt=dt)
    # filtered_signal = cofpf.run(signal.Y)
    # print(complex_to_mag_phase(cofpf.ofpfs[0].get_gain()))
    # print(complex_to_mag_phase(cofpf.ofpfs[1].get_gain()))
    # plt.figure()
    # plt.plot(t, signal.Y[0,:], label=r'$U$')
    # plt.plot(t, filtered_signal.h_hat[0,:], label=r'$\hat U$')
    # plt.legend()

    # plt.figure()
    # plt.plot(t, signal.Y[1,:], label=r'$Y$')
    # plt.plot(t, filtered_signal.h_hat[1,:], label=r'$\hat Y$')
    # plt.legend()

    # plt.show()

    