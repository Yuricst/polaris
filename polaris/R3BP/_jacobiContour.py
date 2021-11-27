#!/usr/bin/env python3
"""
Plotting jacobi contours
Yuri Shimane
"""

import numpy as np
import matplotlib.pyplot as plt


from ._lagrangePoints import lagrangePoints
from ._define_r3bp_param import get_cr3bp_mu


def get_jacobi_contour(
    naifID1,
    naifID2,
    xmin=-1.5,
    xmax=1.5,
    ymin=-1.5,
    ymax=1.5,
    levels=None,
    cmin=2,
    cmax=4,
    figsize=(6, 6),
    cstep=0.05,
    scale=1,
    grid=1000,
    cmap="viridis",
    plot_bodies=True,
    plot_lagrangepoints=False,
):
    """Function creates a plot of Jacobi contour (zero-velocity curve contour)"""
    # non-dimensionalise masses
    mu = get_cr3bp_mu(naifID1, naifID2)
    m1 = 1 - mu
    m2 = mu

    # Lagrange points
    lp = lagrangePoints(mu)

    # create mesh-grid in x-y plane
    x = np.linspace(xmin, xmax, grid)
    y = np.linspace(ymin, ymax, grid)
    z = 0
    [X, Y] = np.meshgrid(x, y)

    # compute potential
    d1 = np.power((X + mu) ** 2 + Y ** 2 + z ** 2, 0.5)
    d2 = np.power((X - 1 + mu) ** 2 + Y ** 2 + z ** 2, 0.5)
    U = 0.5 * (X ** 2 + Y ** 2) + (1 - mu) / d1 + mu / d2

    C = 2 * U  # 0-velocity Jacobi constant

    # define color scale
    if levels is None:
        levels = np.arange(
            cmin, cmax, cstep
        )  # requires fine-tuning (currently tuned for earth-moon)
        use_colorbar = True
    else:
        use_colorbar = False

    ## plot Jacobi contours with Lagrange points
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.contour(X * scale, Y * scale, C, levels, cmap=cmap)
    if use_colorbar:
        fig.colorbar(im)
    ax.clabel(im, inline=True, fontsize=10)
    if plot_bodies:
        ax.scatter(-mu * scale, 0.0, marker="+", s=200, c="b")
        ax.scatter((1 - mu) * scale, 0.0, marker="+", s=200, c="k")
    ax.set(xlabel="x", ylabel="y", title="Zero-velocity contour")
    if plot_lagrangepoints:
        ax.scatter(lp.l1[0] * scale, lp.l1[1] * scale, marker="x", s=120, c="r")
        ax.scatter(lp.l2[0] * scale, lp.l2[1] * scale, marker="x", s=120, c="r")
        ax.scatter(lp.l3[0] * scale, lp.l3[1] * scale, marker="x", s=120, c="r")
        ax.scatter(lp.l4[0] * scale, lp.l4[1] * scale, marker="x", s=120, c="r")
        ax.scatter(lp.l5[0] * scale, lp.l5[1] * scale, marker="x", s=120, c="r")
    #plt.grid(True)
    ax.set_aspect("equal")
    return fig, ax


if __name__ == "__main__":
    print("Hi")
    get_jacobi_contour("399", "301")
