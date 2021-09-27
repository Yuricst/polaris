"""
Test for propagators
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

import sys
sys.path.append('../')   # path to polaris module

import polaris.SolarSystemConstants as sscs
import polaris.Keplerian as kepl
import polaris.Propagator as prop
import polaris.R3BP as r3bp

sys.path.append('../../trajplotlib')
import trajplotlib


if __name__=="__main__":
	state0 = np.array([1, 0, 0, 0, 1, 0])
	tf = 0.863
	mu = 1.0

	propout = prop.propagate_twobody(mu, state0, tf)
	print(propout.statef)