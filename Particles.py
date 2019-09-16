"""
Created on Wed Sep. 10, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

import numpy as np

class Particles(object):
    """Particles:
        Notation Note:
            d: number of states
        Initialize = Particles(self, number_of_particles, X0_range, states_constraints, m):
            number_of_particles: integer, number of particles
            X0_range: 2D list of float with the length d-by-2, range of initial states
            states_constraints: function, constraints on states
            m: integer, number of observations
        Members:
            Np: integer, number of particles
            X: numpy array with the shape of (Np,d), the estimated states for Np particles
            h: numpy array with the shape of (Np,m), the esitmated observations for Np particles
            states_constraints: function, constraints on states
        Methods:
            update(self, dX): use states_constraints function to update states in all particles
    """

    def __init__(self, number_of_particles, X0_range, states_constraints, m):
        self.Np = int(number_of_particles)
        self.X = np.zeros([self.Np, len(X0_range)])
        for d in range(len(X0_range)):
            self.X[:,d] = np.linspace(X0_range[d][0], X0_range[d][1], self.Np)
        self.h = np.zeros([self.Np, m])
        self.states_constraints = states_constraints
        pass

    def update(self, dX):
        self.X = self.states_constraints(self.X+dX)
        pass