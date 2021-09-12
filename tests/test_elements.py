"""
Test for orbital elements
"""

import numpy as np
import sys
sys.path.append('../')   # path to polaris module

import polaris.Keplerian as kepl


def test_elements():
	mu = 398600.44
	r = np.array([6045.0, 1200.0, 0.0])
	v = np.array([-2.024, 6.2, 1.533])
	state = np.concatenate((r, v))
	print(f"State given: \n{state}")

	# convert state-vector to elements
	elts = kepl.sv2elts(state, mu, verbose=True)

	# convert elements back to state-vector
	state_converted = kepl.elts2sv(elts, mu)
	print(f"State converted back: \n{state_converted}")

	# compute difference
	print(f"State after - before conversion: \n{state_converted - state}")


if __name__=="__main__":
	test_elements()

