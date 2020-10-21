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



