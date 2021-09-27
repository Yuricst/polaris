"""
Functions to compute orbital elements
"""

import numpy as np
import numpy.linalg as la
from numba import jit

try:
    from ._rootSolve import __newtonRaphson
except:
    from _rootSolve import __newtonRaphson


@jit(nopython=True)
def get_inclination(state):
    """Function computes inclination in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame

    Returns:
        (float): inclination in radians
    """

    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # inclination
    inc = np.arccos(h[2] / la.norm(h))
    return inc


@jit(nopython=True)
def get_raan(state):
    """Function computes RAAN in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame

    Returns:
        (float): RAAN in radians
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute RAAN
    raan = np.arctan2(ndir[1], ndir[0])
    return raan
    
    
@jit(nopython=True)
def get_eccentricity(state, mu):
    """Function computes eccentricity vector from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (np.arr): eccentricity vector
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    #zdir = np.array([0, 0, 1])
    #ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/la.norm(r)
    return ecc


@jit(nopython=True)
def get_omega(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): argument of periapsis in radians
    """
    
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = np.cross(r,v)
    # normal direction of xy-plane
    zdir = np.array([0, 0, 1])
    ndir = np.cross(zdir, h)
    # compute eccentricity vector
    ecc = (1/mu) * np.cross(v,h) - r/la.norm(r)

    # compute argument of periapsis
    if la.norm(ndir)*la.norm(ecc) != 0.0:
        omega = np.arccos( np.dot(ndir,ecc) / (la.norm(ndir)*la.norm(ecc)) )
        if ecc[2] < 0:
            omega = 2*np.pi - omega
    else:
        omega = 0.0
    return omega


@jit(nopython=True)
def get_trueanom(state, mu):
    """Function computes argument of periapsis in radians from a two-body state vector, in inertially frozen frame
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): true anomaly in radians
    """
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = la.norm( np.cross(r,v) )
    # radial velocity
    vr = np.dot(v,r)/la.norm(r)
    theta = np.arctan2(h*vr, h**2/la.norm(r) - mu)
    return theta


@jit(nopython=True)
def get_semiMajorAxis(state, mu):
    """Function computes semi major axis of keplrian orbit
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter

    Returns:
        (float): semi-major axis
    """
    # decompose state to position and velocity vector
    r = state[0:3]
    v = state[3:6]
    # angular momentum
    h = la.norm( np.cross(r,v) )
    # eccentricity
    e = la.norm( get_eccentricity(state, mu) )
    # semi-major axis
    a = h**2 / (mu*(1 - e**2))
    return a


@jit(nopython=True)
def get_period(state, mu):
    """Function computes period of keplerian orbit
    
    Args:
        state (np.array): array of cartesian state in inertially frozen frame
        mu (float): two-body mass parameter
        
    Returns:
        (float): period
    """
    a = get_semiMajorAxis(state, mu)
    if a < 0:
        period = -1
    else:
        period = 2*np.pi*np.sqrt(a**3/mu)
    return period


@jit(nopython=True)
def eccAnom2meanAnom(eccAnom, ecc):
    meanAnom = eccAnom - ecc*np.sin(eccAnom)
    return meanAnom


# Cannot jit!
def meanAnom2eccAnom(meanAnom, ecc, tol=1.e-13, maxiter=12):
    """Convert mean-anomalty to eccentric anomaly

    """
    eccAnom, convflag = __newtonRaphson(__M2E_res, __M2E_dres, x0=meanAnom, p=(ecc, meanAnom), maxiter=maxiter, tol=tol)
    if convflag is False:
        print("Warning: Newton-Raphson for MA -> EA has not converged!")
    return eccAnom
        

@jit(nopython=True)
def __M2E_res(E, p):
    """meanAnom as a function of eccAnom, for Newton raphson"""
    ecc, M = p
    return E - ecc*np.sin(E) - M


@jit(nopython=True)
def __M2E_dres(E, p):
    """derivative of meanAnom as a function of eccAnom, for Newton raphson"""
    ecc, M = p
    return 1.0 - ecc*np.cos(E)


@jit(nopython=True)
def eccAnom2trueAnom(eccAnom, ecc):
    """Convert eccentric anomaly to true anomaly"""
    ta = 2*np.arctan( np.sqrt((1+ecc)/(1-ecc)) * np.tan(eccAnom/2) )
    return ta 


@jit(nopython=True)
def trueAnom2eccAnom(ta, ecc):
    """Convert true anomaly to eccentric anomaly"""
    eccAnom = 2*np.arctan( np.sqrt((1-ecc)/(1+ecc)) * np.tan(ta/2) )
    return eccAnom

# cannot jit!
def meanAnom2trueAnom(meanAnom, ecc, tol=1.e-13, maxiter=12):
    """Convert mean anomaly to true anomaly"""
    # convert MA to EA
    eccAnom =  meanAnom2eccAnom(meanAnom, ecc, tol=tol, maxiter=maxiter)
    # convert EA to TA
    ta = eccAnom2trueAnom(eccAnom, ecc)
    return __angle_cleanup(ta)

@jit(nopython=True)
def trueAnom2meanAnom(ta, ecc):
    """Convert true anomlay to mean anomlay"""
    # convert TA to EA
    eccAnom = trueAnom2eccAnom(ta, ecc)
    # convert EA to MA
    meanAnom = eccAnom2meanAnom(eccAnom, ecc)
    return __angle_cleanup(meanAnom)

@jit(nopython=True)
def __angle_cleanup(phi, zero_to_2pi=True):
    """Convert angle range

    Args:
        phi (float): angle
        zero_to_2pi (bool): whether to use [0, 2pi] or [-pi, pi]

    Returns:
        (float): angle in specified range
    """
    phi_corrected = phi
    if zero_to_2pi is True:
        if phi < 0.0:
            phi_corrected = 2*np.pi + phi
    else:
        if phi >= np.pi:
            phi_corrected = phi - 2*np.pi
    return phi_corrected