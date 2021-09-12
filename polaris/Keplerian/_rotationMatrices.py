"""
Rotation matrices
"""

import numpy as np
from numba import jit

@jit(nopython=True)
def rotmat1(phi):
	return np.array([ [1.0, 0.0, 0.0],
					  [0.0, np.cos(phi), np.sin(phi)],
					  [0.0, -np.sin(phi), np.cos(phi)] ])

@jit(nopython=True)
def rotmat2(phi):
	return np.array([ [np.cos(phi), 0.0, -np.sin(phi)],
					  [0.0, 1.0, 0.0],
					  [np.sin(phi), 0.0, np.cos(phi)] ])

@jit(nopython=True)
def rotmat3(phi):
	return np.array([ [ np.cos(phi), np.sin(phi), 0.0],
					  [-np.sin(phi), np.cos(phi), 0.0],
					  [0.0, 0.0, 1.0] ])