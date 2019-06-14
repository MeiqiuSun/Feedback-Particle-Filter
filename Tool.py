"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

class Particles(object):
    """Particles:
        Notation Note:
            D: dimenstion of states (X)
        Initialize = Particles(number_of_particles, X0_range, states_constraints, M)
            number_of_particles: integer, number of particles
            X0_range: 2D list of float with length D-by-2, range of initial states
            states_constraints: function, constraints on states
            M: integer, number of outputs
        Members =
            Np: integer, number of particles
            X: 
            h: numpy array with the shape of (Np,M), the esitmated observations for Np particles
            states_constraints: function, constraints on states
        Methods =
            update(): use states_constraints function to update states in all particles
            get_X(): return X (if the dimenstion of states (D) is 1, then the shape of return array is (Np,), otherwise (Np, D))
    """

    def __init__(self, number_of_particles, X0_range, states_constraints, M):
        """Initialize Particles"""
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

if __name__ == "__main__":
    amp_ranges=[[0,1],[0,2],[0,3],[0,4]]
    sigma_B=[0, 1, 2, 3, 4]
    print(amp_ranges)
    print(sigma_B)
    known_observations = [2,3]
    amp_ranges = [amp_ranges[m-1] for m in known_observations]
    known_observations.insert(0,0)
    sigma_B = [sigma_B[m] for m in known_observations]
    print(amp_ranges)
    print(sigma_B)
    pass
    