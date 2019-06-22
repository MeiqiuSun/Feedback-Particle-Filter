"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

import numpy as np

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
    sigma_B=[[0, 1, 2, 3, 4]]
    print(amp_ranges)
    print(sigma_B)
    sigma_B = np.array(sigma_B)
    print(sigma_B, type(sigma_B))
    sigma_B = np.array(sigma_B)
    print(sigma_B, type(sigma_B))
    # known_observations = [2,3]
    # amp_ranges = [amp_ranges[m-1] for m in known_observations]
    # known_observations.insert(0,0)
    # sigma_B = [sigma_B[m] for m in known_observations]
    # print(amp_ranges)
    # print(sigma_B)
    pass
    