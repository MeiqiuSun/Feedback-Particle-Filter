"""
Created on Mon Feb. 25, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def a(X):
    A = np.array([[1,0,1,0],\
                  [0,1,0,1],\
                  [0,0,1,0],\
                  [0,0,0,1]])
    return np.matmul(A,X)

def h(X):
    H = np.array([[0,0,0,1]])
    return np.matmul(H,X)


if __name__ == "__main__":
    X = np.array([1,2,3,4])
    print(a(X))
    print(h(X))
