"""
Clohessy-Wiltshire equations
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def get_stm(t,n=1):
    """Get State-transition matrix
    
    Args:
        t (float): time to obtain STM at time t
        n (float): rotational rate, n = sqrt(mu/r**3)
    """
    # compute sin and cosine
    s = np.sin(n*t)
    c = np.cos(n*t)
    Phi = np.array([
        [4-3*c, 0.0, 0.0, s/n, 2*(1-c)/n, 0.0],
        [6*(s-n*t), 1.0, 0.0, -2*(1-c)/n, (4*s-3*n*t)/n, 0.0],
        [0.0, 0.0, c, 0.0, 0.0, s/n],
        [3*n*s, 0.0, 0.0, c, 2*s, 0.0],
        [-6*n*(1-c), 0.0, 0.0, -2*s, 4*c-3, 0.0],
        [0.0, 0.0, -n*s, 0.0, 0.0, c]
    ])
    return Phi


@jit(nopython=True)
def decompose_stm(Phi):
    """Decompose STM into 3-by-3 blocks"""
    M, N = Phi[0:3,0:3], Phi[0:3, 3:6]
    S, T = Phi[3:6,0:3], Phi[3:6, 3:6]
    return M,N,S,T

#@jit(nopython=True)
def fixed_time_transfer(r0,v0,tf,n=1.0):
    """Compute fixed-time transfer from [r0,v0] to the origin
    
    Args:
        r0 (np.array): initial position in CW frame
        v0 (np.array): initial velocity, in CW frame
        tf (float): transfer time
        n (float): rotational rate, n = sqrt(mu/r**3)
        
    Returns:
        (tuple): dv0, dvf
    """
    Phi = get_stm(tf,n)
    M,N,S,T = decompose_stm(Phi)
    # step 1. drive final position offset to 0
    v0_corrected = -np.dot(np.linalg.pinv(N), np.dot(M,r0))
    dv0 = v0_corrected - v0
    # step 2. find final velocity offset
    dvf = -(np.dot(S,r0) + np.dot(T,v0_corrected))
    return dv0, dvf

