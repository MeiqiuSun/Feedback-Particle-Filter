"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from scipy.signal import find_peaks
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class Particles(object):
    """
    Particles: particles has a scalar state which denotes phase in [0 2pi] cycle.
        Initialize = Particles(number_of_particles, f_min, f_max)
            number_of_particles: integer, number of particles
            f_min: float, minimum frequency in Hz
            f_max: float, maximum frequency in Hz
        Members =
            N: integer, number of particles
            theta: numpy array with the shape of (N,1), the angle (rad) for N particles
            omega_bar: numpy array with the shape of (N,1), the inherited frequency (rad/s) for N particles
            omega: numpy array with the shape of (N,1), the updated frequency (rad/s) for N particles
            amp: numpy array with the shape of (N,1), the amplitude for N particles
            h: numpy array with the shape of (N,1), the esitmated observation for N particles
        Methods =
            update(): mod the angle of each particle with 2pi
    """
    def __init__(self, number_of_particles, f_min, f_max, dt):
        """Initialize Particles"""
        self.dt = dt
        self.N = int(number_of_particles)
        self.theta = np.reshape(np.linspace(0, 2*np.pi, self.N, endpoint=False), [-1,1])
        self.omega_bar = 2*np.pi*np.reshape(np.linspace(f_min, f_max, self.N), [-1,1])
        self.omega = 2*np.pi*np.reshape(np.linspace(f_min, f_max, self.N), [-1,1])
        self.sync_cutoff = np.exp(-((f_max-f_min)/(self.N-1))**2)
        self.amp = np.reshape(np.ones(self.N), [-1,1])
        self.h = np.reshape(np.zeros(self.N), [-1,1])
        self.theta_error_sum = np.reshape(np.zeros(self.N), [-1,1])
        self.Kp = 1
        self.Ki = 0.01
        self.sync_matrix = np.zeros([self.N, self.N])
        self.update_sync_matrix()
    
    def update(self, theta_error):
        """mod the angle of each particle with 2pi"""
        self.theta = np.mod(self.theta, 2*np.pi)
        """update omega"""
        self.theta_error_sum = theta_error*self.dt + 0.9*self.theta_error_sum
        self.omega += self.Kp*theta_error+self.Ki*self.theta_error_sum
        return
        
    def update_sync_matrix(self):
        """update sync_matrix"""
        omega = np.sort(self.omega[:,0])
        for ni in range(self.N):
            for nj in range(ni, self.N):
                self.sync_matrix[ni,nj] = np.exp(-(omega[ni]-omega[nj])**2)
                self.sync_matrix[nj,ni] = self.sync_matrix[ni,nj]
        return self.sync_matrix

    def check_sync(self):
        """check synchronized frequencies"""
        self.update_sync_matrix()
        sync_particles = np.zeros(2*self.N-1)
        for i in range(sync_particles.shape[0]):
            if i < self.N:
                sync_vector = np.zeros(int(i/2)+1)
            else:
                temp_i = 2*self.N-2-i
                sync_vector = np.zeros(int(temp_i/2)+1)
            # Calculate sync_vector
            for j in range(sync_vector.shape[0]):
                sync_vector[j] = self.sync_matrix[int(i/2)+j+np.mod(i,2),int(i/2)-j]    
            # Find the farest sync_particles
            for j in range(sync_vector.shape[0]):
                if sync_vector[j] > self.sync_cutoff:
                    sync_particles[i] =  j
        # Add bound flags at the Head and the Tail
        sync_particles = np.append(-np.insert(sync_particles,0,1),[-1])
        # Filter wripples
        for i in range(sync_particles.shape[0]-2):
            if sync_particles[i] == sync_particles[i+2]:
                sync_particles[i+1] = sync_particles[i]
        
        ## Find Peaks and Cutoff Frequencies
        indices, _ = find_peaks(sync_particles)
        cutoff_sync_freq = np.array([self.omega[int((index+1)/2),0] for index in indices])
        sync_index = np.zeros([cutoff_sync_freq.shape[0]-1,2])
        sync_freq = np.zeros([cutoff_sync_freq.shape[0]-1,2])
        for i in range(sync_freq.shape[0]):
            sync_index[i,:] = (indices[i:i+2]+1)/2
            sync_freq[i,:] = cutoff_sync_freq[i:i+2]

        print(sync_freq/(2*np.pi), sync_index)
        return sync_particles
        # return sync_freq/(2*np.pi), sync_index

    def get_theta(self):
        return np.squeeze(self.theta)
    
    def get_freq(self):
        return np.squeeze(self.omega/(2*np.pi))

    def get_amp(self):
        return np.squeeze(self.amp)

    def get_freq_range(self):
        return np.squeeze(self.omega_bar/(2*np.pi))

class Struct(dict):
    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__.update(kwds)

def isiterable(object):
    try:
        it = iter(object)
    except TypeError: 
        return False
    return True