"""
Kepler-based propagation
"""

import numpy as np 
import numpy.linalg as la 
from numba import jit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from ._statevec2elements import sv2elts, print_elts
from ._elements2statevec import __elts2sv
from ._orbitalElements import trueAnom2meanAnom, meanAnom2trueAnom


def kepler_propagate(mu, state0, t0, tf, tol=1.e-13, maxiter=12):
	"""Propagate using Kepler absed formulation"""
	elts0 = sv2elts(state0, mu)
	print_elts(elts0)

	# get initial mean anomaly
	ta0 = elts0["ta"]
	ma0 = trueAnom2meanAnom(ta0, elts0["ecc"])
	# move forward in time
	maf = ma0 + (2*np.pi/elts0["period"])*tf
	taf = meanAnom2trueAnom(maf, elts0["ecc"], tol, maxiter)
	statef = __elts2sv(sma=elts0["sma"], inc=elts0["ecc"], ecc=elts0["ecc"], 
					 raan=elts0["raan"], aop=elts0["aop"], ta=taf, mu=mu)
	return statef