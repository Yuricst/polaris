#!/usr/bin/env python3
"""
R3BP Stability index for periodic orbits
"""

import numpy as np
import numpy.linalg as la
from numba import jit



# function computes stability index of cr3bp trajectory
#@jit(nopython=True)   # FIXME!
def stabilityIndex(monodromy):
    """Function computes stability index of periodic trajectory in cr3bp
    Args:
        monodromy (np.array): 6x6 numpy array of monodromy matrix
    Returns:
        (float): stability index nu
    """

    # compute eigenvalues and eigenvectors of monodromy
    w, v = la.eig( monodromy )
    # initialize storage
    wreal = np.zeros((6,))
    index = 0
    # iterate over eigenvalues
    for i in range(len(w)):
        if np.imag(w[i])==0:
            wreal[index] = np.real(w[i])
            index += 1

    #if index > 2:
    #    logging.warning('Detected more than 2 real eigenvalues...')
    # compute stability index
    nu = 0.5 * np.abs( wreal[0] + wreal[1] )
    return nu


