"""
Root-solving algorithms
"""

import numpy as np 
from numba import jit


def __newtonRaphson(fun, dfun, x0, p=None, maxiter=12, tol=1.e-13):
    """1D Newton-Raphson algorithm

    Args:
        fun (callable): f(x), signature `fun(x)` or `fun(x, p)`
        dfun (callable): derivative of `f(x)`, signature dfun(x) or `dfun(x, p)`
        x0 (float): initgial guess
        p (list or None): additional parameters
        maxiter (int): max allowed iteration
        tol (float): tolerance
	
	Returns:
		(tuple): x1 and convergence boolean
    """
    convflag = False # initialize
    for idx in range(maxiter):
        if p is None:
            y  = fun(x0) 
            dy = dfun(x0)
        else:
            y  = fun(x0, p)
            dy = dfun(x0, p)
        x1 = x0 - y/dy
        if abs(y) < tol:
            convflag = True
            break
        else:
            x0 = x1
    return x1, convflag
