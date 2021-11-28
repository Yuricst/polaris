"""
Sngle shooting differential corrector
Yuri Shimane
"""


import numpy as np
import numpy.linalg as la
from numba import jit


@jit(nopython=True)
def ssdc(xi, ferr, df):
    """Apply single-shooting differential correction

    Args:
        xi (np.array): length-n array of free variables
        ferr (np.array): length n array of residuals
        df (np.array): n by n np.array of Jacobian
    Returns:
        (np.array): length-n array of new free variables
    """
    xi_vert = np.reshape(xi, (len(xi), -1))
    ferr_vert = np.reshape(ferr, (len(ferr), -1))
    xii_vert = xi_vert - np.dot(la.pinv(df), ferr_vert)
    xii = np.reshape(xii_vert, (len(xii_vert),))
    return xii
