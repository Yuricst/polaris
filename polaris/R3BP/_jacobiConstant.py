#!/usr/bin/env python3
"""
R3BP Jacobi constant
"""

import numpy as np
from numba import jit


# function for Jacobi constant
@jit(nopython=True)
def jacobiConstant(mu,state):
    """Function returns jacobi constant given a cartesian state

    Args:
        mu (float): cr3bp parameter mu
        state (np.array): 1D array of Cartesian state

    Returns:
        (float): Jacobi constant
    """
    
    # unpack state
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt( (x+mu)**2 + y**2 + z**2 )
    r2 = np.sqrt( (x-1+mu)**2 + y**2 + z**2 )
    # compute mass parameters of each primary
    mu1 = 1 - mu
    mu2 = mu
    # compute augmented potential
    ubar = 0.5*(x**2 + y**2) + mu1/r1 + mu2/r2
    jc = -(vx**2 + vy**2 + vz**2) + 2*ubar
    
    return jc


    