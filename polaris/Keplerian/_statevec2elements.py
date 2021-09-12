"""
Functions to convert from state vector to conic elements
"""


import numpy as np

try: 
	from ._orbitalElements import get_inclination, get_raan, get_eccentricity, get_omega, get_trueanom, get_semiMajorAxis, get_period, __angle_cleanup
except:
	from _orbitalElements import get_inclination, get_raan, get_eccentricity, get_omega, get_trueanom, get_semiMajorAxis, get_period, __angle_cleanup


def sv2elts(state, mu, verbose=False):
	"""
	Funciton calculates orbital elements from state vector. 

	Args:
		state (np.array): 6-element state vector, in [km] and [km/s]
		mu (float): gravitational parameter [km^3/s^2]
		
	Returns:
		(dict): dictionary with keys:
			hvec (np.array): angular momentum vector
			incl (float): inclination, in radians
			evec (np.array): eccentricity vector
			raan (float): right-ascension of ascending node, in radians
			aop (float): argument of periapsis, in radians
			ta (float): true anomaly, in radians
	"""
	r = state[0:3]
	v = state[3:6]

	eps = 1.e-15  # singular value handling parameter

	hvec   = np.cross(r,v)
	ecc    = get_eccentricity(state, mu)
	inc    = get_inclination(state)
	raan   = get_raan(state)
	aop    = get_omega(state, mu)
	ta     = get_trueanom(state, mu)
	sma    = get_semiMajorAxis(state, mu)

	# over-write values if necessary
	if (np.abs(r[2]) <= eps) and (np.abs(v[2]) <= eps):
		print("Warning: planar motion detected!")
		inc = 0.0
		raan = 0.0

	if np.linalg.norm(ecc) <= eps:
		print("Warning: circular motion detected!")
		aop = 0.0

	if np.linalg.norm(ecc) < 1.0:
		period = get_period(state, mu)
	else:
		period = -1.0

	# create dictionary to return
	elts_dict = {
		"hvec":   hvec,
		"h":      np.linalg.norm(hvec),
		"inc":    inc,
		"evec":   ecc,
		"ecc":    np.linalg.norm(ecc),
		"raan":   raan, 
		"aop":    aop, 
		"ta":     __angle_cleanup(ta),
		"period": period,
		"sma":    sma
	}
	if verbose is True:
		print_elts(elts_dict)
	return elts_dict


def print_elts(elts_dict):
	"""Print elements stored in dictionary

	Args:
		elts_dict (dict): dictionary containing "sma", inc", "ecc", "raan", "aop", "ta". Angles in radians.
	"""
	# extract from dictionary
	sma  = elts_dict["sma"]
	inc  = elts_dict["inc"]
	ecc  = elts_dict["ecc"]
	raan = elts_dict["raan"]
	aop  = elts_dict["aop"]
	ta   = elts_dict["ta"]
	print("****** ORBITAL ELEMENTS ******")
	print(f"SMA:  {sma:1.6e}")
	print(f"INC:  {inc:1.6e} [rad]")
	print(f"ECC:  {ecc:1.6e}")
	print(f"RAAN: {raan:1.6e} [rad]")
	print(f"AOP:  {aop:1.6e} [rad]")
	print(f"TA:   {ta:1.6e} [rad]")
	print("******************************")
	return