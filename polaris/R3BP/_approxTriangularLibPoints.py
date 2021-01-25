#!/usr/bin/env python3
"""
Linear approximation of triangular libration points
"""

import numpy as np 
from numba import jit

@jit(nopython=True)
def _get_pseudoPotential_deriv(mu, lp=5):
	uxx = 3/4
	uyy = 9/4
	if lp==4:
		uxy =  (3*np.sqrt(mu)/2)*(mu-1/2)
	elif lp==5:
		uxy = -(3*np.sqrt(mu)/2)*(mu-1/2)
	uzz = -1
	return uxx, uyy, uzz, uxy


#@jit(nopython=True)
def get_linearmotion_ic(mu, xi0, eta0, zeta0, lp=5, message=False, period_length="long"):

	# get pseudo-potential derivatives
	uxx, uyy, uzz, uxy = _get_pseudoPotential_deriv(mu, lp)


	# get in-plane motion frequency
	s1 = np.sqrt(-(-1+np.sqrt(1-27*mu+27*mu**2))/2)
	s2 = np.sqrt(-(-1-np.sqrt(1-27*mu+27*mu**2))/2)
	if message==True:
		print(f"s1: {s1}, s2: {s2}")

	# Gamma's
	Gamma1 = (s1**2 + uxx)/(4*s1**2 + uxy**2)
	Gamma2 = (s2**2 + uxx)/(4*s2**2 + uxy**2)
	if message==True:
		print(f"Gamma1: {Gamma1}, Gamma2: {Gamma2}")

	# initial speeds
	if period_length=="short":
		xidot0   = 0.5*(uxy*xi0 + eta0/Gamma2)
		etadot0  = -0.5*((s2**2 + uxx)*xi0 + uxy*eta0)
		period   = 2*np.pi/s2
		theta_ta = np.arctan( 2*Gamma2*uxy/(Gamma2**2*(4*s2**2 + uxy**2) - 1) )/2   # FIXME!

	elif period_length=="long":
		xidot0   = 0.5*(uxy*xi0 + eta0/Gamma1)
		etadot0  = -0.5*((s1**2 + uxx)*xi0 + uxy*eta0)
		period   = 2*np.pi/s1
		theta_ta = np.arctan( 2*Gamma1*uxy/(Gamma1**2*(4*s1**2 + uxy**2) - 1) )/2   # FIXME!

	if message==True:
		print(f"xidot0: {xidot0}, etadot0: {etadot0}, period: {period}")

	return xidot0, etadot0, period, theta_ta


if __name__=='__main__':
	mu = 0.012150585609624
	lstar = 384748.3229297295
	tstar = 375700.34378941957
	xi0   = 384.388174 / lstar
	eta0  = 0.0
	zeta0 = 0.0
	xidot0, etadot0, period, theta_ta = get_linearmotion_ic(mu=mu, xi0=xi0, eta0=eta0, zeta0=zeta0, lp=4, message=True, period_length="long")
	print(f"Dimensional: \n     xidot0:  {xidot0*lstar/tstar:1.8e} [km/s]\n     etadot0: {etadot0*lstar/tstar:1.8e} [km/s]")
	print(f"Period: {period*tstar/86400:2.4f} days")
	print(f"True anomaly: {theta_ta*180/np.pi:2.8f} deg")

