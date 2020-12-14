#!/usr/bin/env python3
"""
Functions to convert from state vector to conic elements
"""


import numpy as np
from ._conicElements import get_inclination, get_raan, get_eccentricity, get_omega, get_trueanom, get_semiMajorAxis, get_period



def sv2elts(state, mu):
	"""
	Funciton calculates orbital elements from state vector for e<1

	Args:
		state (np.array): 6-element state vector, in [km] and [km/s]
		mu (float): gravitational parameter [km^3/s^2]
		
	Returns:
		(dict): dictionary with keys:
			hvec (np.array): angular momentum vector
			incl (float): inclination, in radians
			evec (np.array): eccentricity vector
			raan (float): right-ascension of ascending node, in radians
			omega (float): argument of periapsis, in radians
			ta (float): true anomaly, in radians
	"""
	r = state[0:3]
	v = state[3:6]
	hvec   = np.cross(r,v)
	incl   = get_inclination(state)
	raan   = get_raan(state)
	ecc    = get_eccentricity(state, mu)
	omega  = get_omega(state, mu)
	theta  = get_trueanom(state, mu)
	period = get_period(state, mu)
	a      = get_semiMajorAxis(state, mu)

	# create dictionary to return
	elts = {
		"hvec": hvec,
		"incl": incl,
		"evec": ecc,
		"raan": raan, 
		"omega": omega, 
		"theta": theta,
		"period": period,
		"semiMajorAxis": a
	}
	return elts


