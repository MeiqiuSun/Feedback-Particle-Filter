"""
Created on Fri Feb. 15, 2019

@author: Heng-Sheng (Hanson) Chang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import bisect

class Struct(dict):
    """create a struct"""
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__.update(kwargs)

def isiterable(object):
    """check if the object is iterable"""
    try:
        iter(object)
    except TypeError: 
        return False
    return True

def find_limits(signals, scale='normal'):
    """find the maximum and minimum of plot limits
        Notation Note = 
            M: number of signals
            T: length of a signal
        Parameters = 
            signals: numpy array with the shape of (M,T), signal array
    """
    maximums = np.zeros(signals.shape[0])
    minimums = np.zeros(signals.shape[0])
    for m in range(signals.shape[0]):
        maximums[m] = np.max(signals[m])
        minimums[m] = np.min(signals[m])

    if scale=='log':
        signal_range = np.log10(np.max(maximums))-np.log10(np.min(minimums))
        minimum = np.power(10, np.log10(np.min(minimums))-signal_range/100)
        maximum = np.power(10, np.log10(np.max(maximums))+signal_range/100)
    else:
        signal_range = np.max(maximums)-np.min(minimums)
        minimum = np.min(minimums)-signal_range/10
        maximum = np.max(maximums)+signal_range/10
    return minimum, maximum

def find_closest(num, seq, pos=False):
    """Find the closest number(num) in a sequence(seq). Return the value or the position."""
    index = bisect.bisect_left(a=seq, x=num)
    if index == len(seq):
        raise ValueError
    if pos:
        return index
    else :
        return seq[index]

def complex_to_mag_phase(gain):
    # gain is a numpy complex array with the shape of (2,)
    mag = np.abs(gain)
    phase = np.angle(gain)
    while phase[1]-phase[0] > 0:
        phase[0] += 2*np.pi
    return mag[1]/mag[0], phase[1]-phase[0]

default_colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
    ]

if __name__ == "__main__":
    pass
    