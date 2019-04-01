"""
Created on Wed Mar. 27, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class KF(object):
    def __init__(self, model, sigma_B, sigma_W, dt):
        self.sigma_B = model.modified_sigma_B(np.array(sigma_B))
        self.sigma_W = model.modified_sigma_W(np.array(sigma_W))
        self.N = len(self.sigma_B)
        self.M = len(self.sigma_W)
        self.dt = dt
        self.f = model.f
        self.h = model.h

    def update(self, y):
        


