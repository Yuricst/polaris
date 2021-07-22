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
	# system parameter
	EarthMoonSystem = r3bp.CR3BP(399, 301)
	EarthMoonSunSystem = r3bp.BCR4BP(399, 301)

	# initial state
	tf = 3.4142341769218727 / 2
	state0 = np.array([1.11978627, 0.0, 0.00910361, 0.0, 0.17778276, 0.0])

	# propagate in CR3BP
	prop_cr3bp = prop.propagate_cr3bp(EarthMoonSystem, state0, tf, force_solve_ivp=True)
	ax = trajplotlib.quickplot3(prop_cr3bp.xs, prop_cr3bp.ys, prop_cr3bp.zs, c_traj="navy")

	# propagate in BCR4BP
	beta0 = 0.2
	prop_bcr4bp = prop.propagate_bcr4bp(EarthMoonSunSystem, state0, tf, beta0, force_solve_ivp=True)
	ax = trajplotlib.quickplot3(prop_bcr4bp.xs, prop_bcr4bp.ys, prop_bcr4bp.zs, ax=ax, c_traj="deeppink")

	plt.show()