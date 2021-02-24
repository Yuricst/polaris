#!/usr/bin/env python3
"""
Miscellaneous functions for computing parameters in Keplerian dynamics
"""

import numpy as np
import numpy.linalg as la
from numba import jit



@jit(nopython=True)
def get_synodic_period(p1, p2):
    """Compute synodic period between two systems with periods p1 and p2

    Args:
        p1 (float): period of first system
        p2 (float): period of second system

    Returns:
        (float): synodic period
    """
    return 1/np.abs( 1/p1 - 1/p2 )


@jit(nopython=True)
def get_energy(mu, r, v):
    """Compute specific energy

    Args:
        mu (float): gravitational parameter
        r (float): radius, scalar
        v (float): velocity, scalar

    Returns:get_
        (float): c3
    """
    return 0.5*v**2 - mu/r


@jit(nopython=True)
def get_c3(mu, r, v):
    """Compute characteristic energy C3

    Args:
        mu (float): gravitational parameter
        r (float): radius, scalar
        v (float): velocity, scalar

    Returns:
        (float): c3
    """
    return 2*get_energy(mu, r, v)


@jit(nopython=True)
def get_fpa(state, mu):
    """Get flight-path angle, between perpendicular to position vector and velocity vector
    
    Args:
        state (np.array): state-vector
        mu (float): gravitational parameter
    
    Returns:
        (float): FPA, in radians
        
    """
    h = la.norm(np.cross(state[0:3], state[3:6]))
    vperp = h / la.norm(state[0:3])
    vr = mu/h * la.norm(get_eccentricity(state, mu)) * np.sin(get_trueanomaly(state, mu))
    gamma = np.arctan(vr / vperp)
    return gamma


def get_hohmann_cost(mu, r1, r2):
    """Compute Hohmann transfer cost

    Args:
        mu
        r1
        r2

    Returns:
        (dict): dictionary containing Hohmann transfer parameters
    """
    # semi-major axis
    ahto = (r1+r2)/2
    # circular velocities at r1 and r2
    vcirc_1 = np.sqrt(mu/r1)
    vcirc_2 = np.sqrt(mu/r2)
    # time of flight
    tof = np.pi*np.sqrt(ahto**3/mu)
    # hto velocity
    v1 = np.sqrt(mu*(2/r1 - 1/ahto))
    v2 = np.sqrt(mu*(2/r2 - 1/ahto))
    # dv cost
    dv1 = np.abs(v1 - vcirc_1)
    dv2 = np.abs(v2 - vcirc_2)
    # print information
    print(f"Hohmann transfer summary:")
    print(f"  tof: {tof:1.4e}")
    print(f"  r1: {r1:1.4e}, vcirc_1: {vcirc_1:1.4e}, vhto_1: {v1:1.4e}, dv_1: {dv1:1.4e}")
    print(f"  r2: {r2:1.4e}, vcirc_1: {vcirc_2:1.4e}, vhto_2: {v2:1.4e}, dv_2: {dv2:1.4e}")
    # prepare dictionary
    hohmann = {
        "r1": r1,
        "r2": r2,
        "vcirc_1": vcirc_1,
        "vcirc_2": vcirc_2,
        "vhto1": v1, 
        "vhto2": v2,
        "dv1": dv1,
        "dv2": dv2,
        "tof": tof,
    }
    return hohmann