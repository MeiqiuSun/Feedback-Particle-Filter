"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

class Particles(object):
    """Particles:
        Initialize = Particles(number_of_particles, X0_range, states_constraints, M, dt)
            number_of_particles: integer, number of particles
        Members =
            Np: integer, number of particles
            h: numpy array with the shape of (Np,1), the esitmated observation for Np particles
        Methods =
            update(): mod the angle of each particle with 2pi and update omega with PI controller
            update_sync_matrix(): update synchronized matrix
            check_sync(): check synchronized frequencies
            get_theta(): return a numpy array with the shape of (Np,) of angle (rad) for Np particles
            get_freq(): return a numpy array with the shape of (Np,) of frequency (Hz) for Np particles
            get_amp: return a numpy array with the shape of (Np,) of amplitude for Np particles
            get_freq_range(): return a numpy array with the shape of (Np,) of frequency range (Hz) for Np particles
    """

    def __init__(self, number_of_particles, X0_range, states_constraints, M, dt):
        """Initialize Particles"""
        self.dt = dt
        self.Np = int(number_of_particles)
        # initialize states (X)
        self.X = np.zeros([self.Np, len(X0_range)])
        for n in range(len(X0_range)):
            self.X[:,n] = np.linspace(X0_range[n][0], X0_range[n][1], self.Np, endpoint=False)
        self.h = np.zeros([self.Np, M])
        self.states_constraints = states_constraints

    def update(self):
        self.X = self.states_constraints(self.X)
        return 

    def get_X(self):
        return np.squeeze(self.X)

class Struct(dict):
    """create a struct"""
    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__.update(kwds)

def isiterable(object):
    """check if object is iterable"""
    try:
        iter(object)
    except TypeError: 
        return False
    return True

def find_limits(signals, scale='normal'):
    """find the maximum and minimum of plot limits"""
    maximums = np.zeros(signals.shape[0])
    minimums = np.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        maximums[i] = np.max(signals[i])
        minimums[i] = np.min(signals[i])

    if scale=='log':
        signal_range = np.log10(np.max(maximums))-np.log10(np.min(minimums))
        minimum = np.power(10, np.log10(np.min(minimums))-signal_range/100)
        maximum = np.power(10, np.log10(np.max(maximums))+signal_range/100)
    else:
        signal_range = np.max(maximums)-np.min(minimums)
        minimum = np.min(minimums)-signal_range/10
        maximum = np.max(maximums)+signal_range/10
    return minimum, maximum