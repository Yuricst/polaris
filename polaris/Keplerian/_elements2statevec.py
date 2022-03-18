"""
Convert orbital elements to state-vector
"""

import numpy as np
import numpy.linalg as la
from numba import jit

try:
    from ._rotationMatrices import rotmat1, rotmat3
except:
    from _rotationMatrices import rotmat1, rotmat3


def elts2sv(elts_dict, mu):
    """
    Convert keplerian elements to state-vector

    Args:
        elts_dict (dict): dictionary containing "sma", inc", "ecc", "raan", "aop", "ta". Angles in radians.
        mu (float): gravitational parameter

    Returns:
        (np.array): state-vector
    """
    # extract from dictionary
    sma  = elts_dict["sma"]
    inc  = elts_dict["inc"]
    ecc  = elts_dict["ecc"]
    raan = elts_dict["raan"]
    aop  = elts_dict["aop"]
    ta   = elts_dict["ta"]
    assert ecc < 1.0
    # run jit-ed function
    return __elts2sv(sma, inc, ecc, raan, aop, ta, mu)



@jit(nopython=True)
def __elts2sv(sma, inc, ecc, raan, aop, ta, mu):
    # construct state in perifocal frame
    h = np.sqrt(sma*mu*(1 - ecc**2))
    rPF = h**2 / (mu*(1 + ecc*np.cos(ta))) * np.array([np.cos(ta), np.sin(ta), 0.0])
    vPF = mu/h * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])
    # convert to inertial frame
    if (aop != 0.0):
        r1 = np.dot(rotmat3(-aop), rPF)
        v1 = np.dot(rotmat3(-aop), vPF)
    else:
        r1 = rPF
        v1 = vPF

    if (inc != 0.0):
        r2 = np.dot(rotmat1(-inc), r1)
        v2 = np.dot(rotmat1(-inc), v1)
    else:
        r2 = r1
        v2 = v1
    
    if (raan != 0.0):
        rI = np.dot(rotmat3(-raan), r2)
        vI = np.dot(rotmat3(-raan), v2)
    else:
        rI = r2
        vI = v2
    return np.concatenate((rI, vI))