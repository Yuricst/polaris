#!/usr/bin/env python3
"""
Plotting jacobi contours
Yuri Shimane
"""

import numpy as np 
import matplotlib.pyplot as plt


from ._lagrangePoints import lagrangePoints
from ._define_r3bp_param import get_cr3bp_mu


def get_jacobi_contour(naifID1, naifID2):

	# non-dimensionalise masses
	mu = get_cr3bp_mu(naifID1, naifID2)
	m1 = 1 - mu
	m2 = mu

	# Lagrange points
	lp = lagrangePoints(mu)

	# create mesh-grid in x-y plane
	grid = 1000
	x = np.linspace(-1.5,1.5,grid)
	y = np.linspace(-1.5,1.5,grid)
	z = 0
	[X,Y] = np.meshgrid(x,y)

	# compute potential
	d1 = np.power((X + mu)**2 + Y**2 + z**2, 0.5)
	d2 = np.power((X - 1 + mu)**2 + Y**2 + z**2, 0.5)
	U = 0.5*(X**2 + Y**2)+(1 - mu)/d1 + mu/d2

	C = 2*U # 0 velocity Jacobi constant

	# define color scale
	cc = [2:0.05:3.2, 2:0.2:6]  # requires fine-tuning (currently tuned for earth-moon)


	## plot Jacobi contours with Lagrange points
	plt.rcParams["font.size"] = 20
	fig, ax = plt.subplots(1,1, figsize=(12,8))

	ax.contour(X, Y, C, cc, cmap='RdGy')

	# L1
	# plot(lp(1,1),lp(1,2),'xr')
	# text(lp(1,1),lp(1,2)-0.15,'L1','FontSize',16)
	# # L2
	# plot(lp(2,1),lp(2,2),'xr')
	# text(lp(2,1),lp(2,2)-0.15,'L2','FontSize',16)
	# # L3
	# plot(lp(3,1),lp(3,2),'xr')
	# text(lp(3,1),lp(3,2)-0.15,'L3','FontSize',16)
	# # L4
	# plot(lp(4,1),lp(4,2),'xr')
	# text(lp(4,1),lp(4,2)-0.15,'L4','FontSize',16)
	# # L5
	# plot(lp(5,1),lp(5,2),'xr')
	# text(lp(5,1),lp(5,2)-0.15,'L5','FontSize',16)
	# title(['Jacobi constant contour for mu = ',num2str(mu)])

	ax.set(xlabel='x, canonical', ylabel='y, canonical', title='ZVC')
	plt.grid(True)
	plt.axis("equal")
	plt.show()


if __name__ == "__main__":
	print('Hi')
	get_jacobi_contour("399", "301")

