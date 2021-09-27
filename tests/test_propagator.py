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
	period = 3.4142341769218727
	tf = period / 2
	state0 = np.array([1.11978627, 0.0, 0.00910361, 0.0, 0.17778276, 0.0])

	# propagate in CR3BP
	print("Testing CR3BP propagation...")
	prop_cr3bp = prop.propagate_cr3bp(EarthMoonSystem, state0, tf, force_solve_ivp=True)
	ax = trajplotlib.quickplot3(prop_cr3bp.xs, prop_cr3bp.ys, prop_cr3bp.zs, c_traj="navy")
	print("Success!")

	# propagate in BCR4BP
	print("Testing BCR4BP propagation...")
	beta0 = 0.2
	prop_bcr4bp = prop.propagate_bcr4bp(EarthMoonSunSystem, state0, tf, beta0, force_solve_ivp=True)
	ax = trajplotlib.quickplot3(prop_bcr4bp.xs, prop_bcr4bp.ys, prop_bcr4bp.zs, ax=ax, c_traj="deeppink")
	print("Success!")

	# construct manifold
	print("Testing manifold...")
	manif_pls, manif_min = r3bp.get_manifold(EarthMoonSystem, state0, period, 4.0)
	print("Success!")
	
	for idx, branch in enumerate(manif_pls.branches):
		if idx==0:
			ax2 = trajplotlib.quickplot2(branch.propout.xs, branch.propout.ys, n_figsize=7,
				scale=1.4, c_traj="#ffb3f0", c_start="#9b5348", c_end="#6f0043",
				marker_start=None, marker_end=None)
		else:
			ax2 = trajplotlib.quickplot2(branch.propout.xs, branch.propout.ys, 
				scale=1.4, ax=ax2, c_traj="#ffb3f0", c_start="#9b5348", c_end="#6f0043",
				marker_start=None, marker_end=None)
	plt.show()

