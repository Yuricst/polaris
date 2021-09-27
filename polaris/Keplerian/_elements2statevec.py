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
		mu (float): gravitational parameter
		elts_dict (dict): dictionary containing "sma", inc", "ecc", "raan", "aop", "ta". Angles in radians.

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
	if (raan != 0.0) and (inc != 0.0) and (aop != 0.0):
		rI = np.dot(rotmat3(-raan), np.dot(rotmat1(-inc), np.dot(rotmat3(-aop), rPF)))
		vI = np.dot(rotmat3(-raan), np.dot(rotmat1(-inc), np.dot(rotmat3(-aop), vPF)))
	elif (raan != 0.0) and (inc != 0.0) and (aop == 0.0):
		rI = np.dot(rotmat3(-raan), np.dot(rotmat1(-inc), rPF))
		vI = np.dot(rotmat3(-raan), np.dot(rotmat1(-inc), vPF))
	elif (raan != 0.0) and (inc == 0.0) and (aop == 0.0):
		rI = np.dot(rotmat3(-raan), rPF)
		vI = np.dot(rotmat3(-raan), vPF)
	else:
		rI = rPF
		vI = vPF
	return np.concatenate((rI, vI))