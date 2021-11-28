#!/usr/bin/env python3
"""
R3BP lagrange points (libration points)
"""

import numpy as np
from scipy import optimize

# function (and return class) for computing Lagrange points
class _lagrangePointsReturn:
    def __init__(self, l1, l2, l3, l4, l5):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5


def lagrangePoints(mu):
    """Function computes location of Lagrange points
    Args:
        mu (float): CR3BP system parameter mu
    Returns:
        (struct): structure with self.l1 ~ self.l5 numpy arrays of x,y,z coordinates of Lagrange point
    """

    # define l = 1-mu
    l = 1 - mu

    # collinear points
    def eqL1(x):
        fval = (
            x ** 5
            + 2 * (mu - l) * x ** 4
            + (l ** 2 - 4 * l * mu + mu ** 2) * x ** 3
            + (2 * mu * l * (l - mu) + mu - l) * x ** 2
            + (mu ** 2 * l ** 2 + 2 * (l ** 2 + mu ** 2)) * x
            + mu ** 3
            - l ** 3
        )
        # fval = gamma**5 - (3-mu)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 + 2*mu*gamma - mu
        return fval

    sol_l1 = optimize.root(eqL1, 0.5, method="hybr")
    l1 = np.array([sol_l1.x[0], 0, 0])

    def eqL2(x):
        fval = (
            x ** 5
            + 2 * (mu - l) * x ** 4
            + (l ** 2 - 4 * l * mu + mu ** 2) * x ** 3
            + (2 * mu * l * (l - mu) - (mu + l)) * x ** 2
            + (mu ** 2 * l ** 2 + 2 * (l ** 2 - mu ** 2)) * x
            - (mu ** 3 + l ** 3)
        )
        # fval = gamma**5 + (3-mu)*gamma**4 + (3-2*mu)*gamma**3 - mu*gamma**2 - 2*mu*gamma - mu
        return fval

    sol_l2 = optimize.root(eqL2, 1.5, method="hybr")
    l2 = np.array([sol_l2.x[0], 0, 0])

    def eqL3(x):
        fval = (
            x ** 5
            + 2 * (mu - l) * x ** 4
            + (l ** 2 - 4 * mu * l + mu ** 2) * x ** 3
            + (2 * mu * l * (l - mu) + (l + mu)) * x ** 2
            + (mu ** 2 * l ** 2 + 2 * (mu ** 2 - l ** 2)) * x
            + l ** 3
            + mu ** 3
        )
        return fval

    sol_l3 = optimize.root(eqL3, -1, method="hybr")
    l3 = np.array([sol_l3.x[0], 0, 0])

    # equilateral points
    # L4
    l4 = np.array([np.cos(np.pi / 3) - mu, np.sin(np.pi / 3), 0])
    # L5
    l5 = np.array([np.cos(np.pi / 3) - mu, -np.sin(np.pi / 3), 0])

    return _lagrangePointsReturn(l1, l2, l3, l4, l5)
