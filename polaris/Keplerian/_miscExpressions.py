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
