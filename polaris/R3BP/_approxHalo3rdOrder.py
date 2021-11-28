#!/usr/bin/env python3
"""
Function for generating analytical 3rd order approximation of halo
Yuri Shimane, 2020.11.07
"""


import numpy as np
from scipy import optimize
from scipy.optimize import minimize


from ._lagrangePoints import lagrangePoints


# -------------------------------------------------------------------------------------------------------- #
def compute_legendre_const(n, mu, gammaL, lp):
    """Function computes Legendre constant c_n for collinear CR3BP system"""
    if lp == 1:
        out = (1 / gammaL ** 3) * (
            mu + (-1) ** n * (1 - mu) * gammaL ** (n + 1) / (1 - gammaL) ** (n + 1)
        )
    elif lp == 2:
        out = (1 / gammaL ** 3) * (
            (-1) ** n * mu
            + (-1) ** n * (1 - mu) * gammaL ** (n + 1) / (1 + gammaL) ** (n + 1)
        )
    elif lp == 3:
        out = (1 / gammaL ** 3) * (
            1 - mu + mu * gammaL ** (n + 1) / (1 + gammaL) ** (n + 1)
        )
    return out


# -------------------------------------------------------------------------------------------------------- #
def linear_cr3bp_lambda(c2):
    """Function computes roots of characteristic equation for coupled XY-motion"""

    # solve roots of equation:  fval =  lmb**4 + (c2-2)*lmb**2 - (c2 - 1)*(1 + 2*c2)
    bb = c2 - 2
    cc = -(c2 - 1) * (1 + 2 * c2)

    # solve for lambda^2
    lmb_sqrd_pls = (-bb + np.sqrt(bb ** 2 - 4 * cc)) / 2
    lmb_sqrd_min = (-bb - np.sqrt(bb ** 2 - 4 * cc)) / 2

    # four lambda solutions
    if lmb_sqrd_pls > 0 and lmb_sqrd_min < 0:
        lmb_sol = np.abs(np.sqrt(lmb_sqrd_pls))
    elif lmb_sqrd_pls < 0 and lmb_sqrd_min > 0:
        lmb_sol = np.abs(np.sqrt(lmb_sqrd_min))
    else:
        lmb_sol = 1.0
    return lmb_sol


# -------------------------------------------------------------------------------------------------------- #
def compute_legendre_coeff(c2, c3, c4, lmb, k):
    """Function computes Legendre coefficients"""

    # coefficient d1, d2
    d1 = (3 * (lmb ** 2) / k) * (k * (6 * lmb ** 2 - 1) - 2 * lmb)
    d2 = (8 * (lmb ** 2) / k) * (k * (11 * lmb ** 2 - 1) - 2 * lmb)
    # coefficient d21
    d21 = -c3 / (2 * lmb ** 2)
    # coefficient b21, b22
    b21 = (-3 * c3 * lmb / (2 * d1)) * (3 * k * lmb - 4)
    b22 = 3 * c3 * lmb / d1
    # coefficient a21 ~ a24
    a21 = 3 * c3 * (k ** 2 - 2) / (4 * (1 + 2 * c2))
    a22 = 3 * c3 / (4 * (1 + 2 * c2))
    a23 = (-3 * c3 * lmb / (4 * k * d1)) * (3 * k ** 3 * lmb - 6 * k * (k - lmb) + 4)
    a24 = (-3 * c3 * lmb / (4 * k * d1)) * (2 + 3 * k * lmb)
    # b31 ~ b32
    b31 = (3 / (8 * d2)) * (
        8 * lmb * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k ** 2))
        + (9 * lmb ** 2 + 1 + 2 * c2)
        * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2))
    )
    b32 = (1 / d2) * (
        9 * lmb * (c3 * (k * b22 + d21 - 2 * a24) - c4)
        + (3 / 8) * (9 * lmb ** 2 + 1 + 2 * c2) * (4 * c3 * (k * a24 - b22) + k * c4)
    )
    # a31 ~ a32
    a31 = (-9 * lmb / (4 * d2)) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k ** 2)) + (
        (9 * lmb ** 2 + 1 - c2) / (2 * d2)
    ) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k ** 2))
    a32 = (-1 / d2) * (
        (9 * lmb / 4) * (4 * c3 * (k * a24 - b22) + k * c4)
        + (3 / 2) * (9 * lmb ** 2 + 1 - c2) * (c3 * (k * b22 + d21 - 2 * a24) - c4)
    )
    # coefficient d31 ~ d32
    d31 = (3 / (64 * lmb ** 2)) * (4 * c3 * a24 + c4)
    d32 = (3 / (64 * lmb ** 2)) * (4 * c3 * (a23 - d21) + c4 * (4 + k ** 2))
    # frequency correction s1, s2
    s1 = (1 / (2 * lmb * (lmb * (1 + k ** 2) - 2 * k))) * (
        (3 / 2) * c3 * (2 * a21 * (k ** 2 - 2) - a23 * (k ** 2 + 2) - 2 * k * b21)
        - (3 / 8) * c4 * (3 * k ** 4 - 8 * k ** 2 + 8)
    )
    s2 = (1 / (2 * lmb * (lmb * (1 + k ** 2) - 2 * k))) * (
        (3 / 2)
        * c3
        * (2 * a22 * (k ** 2 - 2) + a24 * (k ** 2 + 2) + 2 * k * b22 + 5 * d21)
        + (3 / 8) * c4 * (12 - k ** 2)
    )
    # amplitude constraint param. a1, a2, l1, l2
    a1 = -(3 / 2) * c3 * (2 * a21 + a23 + 5 * d21) - (3 / 8) * c4 * (12 - k ** 2)
    a2 = (3 / 2) * c3 * (a24 - 2 * a22) + (9 / 8) * c4
    l1 = a1 + 2 * lmb ** 2 * s1
    l2 = a2 + 2 * lmb ** 2 * s2

    return (
        a21,
        a22,
        a23,
        a24,
        a31,
        a32,
        b21,
        b22,
        b31,
        b32,
        d21,
        d31,
        d32,
        d1,
        d2,
        a1,
        a2,
        s1,
        s2,
        l1,
        l2,
    )


# -------------------------------------------------------------------------------------------------------- #
def get_halo_approx(mu, lp, lstar, az_km, family=1, phase=0.0, message=False):
    """Function returns approximate state and period of collinear halo in 3rd-order approximation

    Args:
        mu (float): CR3BP system parameter
        lp (int): lagrange point about which to construct halo; 1, 2, or 3
        lstar (float): canonical length scale of CR3BP system
        az_km (float): max z-direction amplitude, in km
        family (int): North or South family (1 or 3); whether the resulting trajectory is North or South depends on lp
        phase (float): phase of trajectory to construct the initial guess state; use 0 or pi in order to correct the initial guess to numerical periodic solution later
        message (bool): whether to display messages

    Returns:
        (dict): dictionary containing initial guess solution of the state and the period, as well as the 3rd order periodic motion's states over one full period; keys are:
            "times", "xs", "ys", "zs", "vxs", vys", "vzs", "state_guess", "period_guess"
    """

    # compute lagrange points
    lpoint = lagrangePoints(mu)

    # compute gamma
    if lp == 1:
        gammaL = np.abs((1 - mu) - lpoint.l1[0])
    elif lp == 2:
        gammaL = np.abs((1 - mu) - lpoint.l2[0])
    elif lp == 3:
        gammaL = np.abs(mu - lpoint.l3[0])
    else:
        raise Exception(f"Lagrange point must be 1, 2, or 3 (set with argument lp)")

    # compute constants
    c2 = compute_legendre_const(2, mu, gammaL, lp)
    c3 = compute_legendre_const(3, mu, gammaL, lp)
    c4 = compute_legendre_const(4, mu, gammaL, lp)

    if message == True:
        print(f"gammaL: {gammaL}, c2:{c2}, c3:{c3}, c4: {c4}")

    # compute linear cr3bp motion angular speed lambda
    lmb = linear_cr3bp_lambda(c2)

    # compute frequency correction term Delta
    Delta = lmb ** 2 - c2

    # compute k
    k = 2 * lmb / (lmb ** 2 + 1 - c2)

    # compute legendre coefficients
    (
        a21,
        a22,
        a23,
        a24,
        a31,
        a32,
        b21,
        b22,
        b31,
        b32,
        d21,
        d31,
        d32,
        d1,
        d2,
        a1,
        a2,
        s1,
        s2,
        l1,
        l2,
    ) = compute_legendre_coeff(c2, c3, c4, lmb, k)

    # compute amplitudes
    Az = az_km / (lstar * gammaL)
    Ax = np.sqrt((-Delta - l2 * Az ** 2) / l1)
    Ay = k * Ax
    if message == True:
        print(
            f"Ax = {Ax*lstar*gammaL:.2f} [km], Ay = {Ay*lstar*gammaL:.2f} [km], Az = {Az*lstar*gammaL:.2f} [km]"
        )

    # halo orbit (analytical) period
    omega1 = 0
    omega2 = s1 * (Ax) ** 2 + s2 * (Az) ** 2
    omega = 1 + omega1 + omega2
    period_richard = 2 * np.pi / (lmb * omega)  # -- non-dimensionalized by 1/Tstar
    if message == True:
        print(f"Analytical period = {period_richard:.8f} (non-dim)")

    # third-order time array
    times = omega * np.linspace(
        0, period_richard, 500
    )  # time variable for 3rd order solution

    # switching function
    deltan = 2 - family
    # initialize
    x_analytic_Lframe = np.zeros(len(times))
    y_analytic_Lframe = np.zeros(len(times))
    z_analytic_Lframe = np.zeros(len(times))
    xdot_analytic_Lframe = np.zeros(len(times))
    ydot_analytic_Lframe = np.zeros(len(times))
    zdot_analytic_Lframe = np.zeros(len(times))

    # third order solution in Libration-frame
    for i in range(len(times)):
        # positions in Lframe
        x_analytic_Lframe[i] = (
            a21 * Ax ** 2
            + a22 * Az ** 2
            - Ax * np.cos(lmb * times[i] + phase)
            + (a23 * Ax ** 2 - a24 * Az ** 2) * np.cos(2 * (lmb * times[i] + phase))
            + (a31 * Ax ** 3 - a32 * Ax * Az ** 2)
            * np.cos(3 * (lmb * times[i] + phase))
        )

        y_analytic_Lframe[i] = (
            k * Ax * np.sin(lmb * times[i] + phase)
            + (b21 * Ax ** 2 - b22 * Az ** 2) * np.sin(2 * (lmb * times[i] + phase))
            + (b31 * Ax ** 3 - b32 * Ax * Az ** 2)
            * np.sin(3 * (lmb * times[i] + phase))
        )

        z_analytic_Lframe[i] = (
            deltan * Az * np.cos(lmb * times[i] + phase)
            + deltan * d21 * Ax * Az * (np.cos(2 * (lmb * times[i] + phase)) - 3)
            + deltan
            * (d32 * Az * Ax ** 2 - d31 * Az ** 3)
            * np.cos(3 * (lmb * times[i] + phase))
        )

        # velocities
        xdot_analytic_Lframe[i] = (
            lmb * Ax * np.sin(lmb * times[i] + phase)
            + (a23 * Ax ** 2 - a24 * Az ** 2)
            * 2
            * lmb
            * np.sin(2 * (lmb * times[i] + phase))
            - (a31 * Ax ** 3 - a32 * Ax * Az ** 2)
            * 3
            * lmb
            * np.sin(3 * (lmb * times[i] + phase))
        )

        ydot_analytic_Lframe[i] = (
            k * Ax * lmb * omega * np.cos(lmb * times[i] + phase)
            + (b21 * Ax ** 2 - b22 * Az ** 2)
            * 2
            * lmb
            * omega
            * np.cos(2 * (lmb * times[i] + phase))
            + (b31 * Ax ** 3 - b32 * Ax * Az ** 2)
            * 3
            * lmb
            * omega
            * np.cos(3 * (lmb * times[i] + phase))
        )

        zdot_analytic_Lframe[i] = (
            -deltan * Az * lmb * np.sin(lmb * times[i] + phase)
            - deltan * d21 * Ax * Az * 2 * lmb * np.sin(2 * (lmb * times[i] + phase))
            - deltan
            * (d32 * Az * Ax ** 2 - d31 * Az ** 3)
            * 3
            * lmb
            * np.sin(3 * (lmb * times[i] + phase))
        )

    # ... third order analytical solution in synodic frame
    if lp == 1:
        xs = gammaL * x_analytic_Lframe + lpoint.l1[0]
    elif lp == 2:
        xs = gammaL * x_analytic_Lframe + lpoint.l2[0]
    elif lp == 3:
        xs = gammaL * x_analytic_Lframe + lpoint.l3[0]
    ys = gammaL * y_analytic_Lframe
    zs = gammaL * z_analytic_Lframe
    vxs = gammaL * xdot_analytic_Lframe
    vys = gammaL * ydot_analytic_Lframe
    vzs = gammaL * zdot_analytic_Lframe

    # extract as initial analytical solution
    state0 = np.array(
        [xs[0], 0.0, zs[0], 0.0, vys[0], 0.0]
    )  # forcefully turning y, vx, vz to 0

    # construct output dictionary
    out = {
        "times": times,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "vxs": vxs,
        "vys": vys,
        "vzs": vzs,
        "state_guess": state0,
        "period_guess": period_richard,
    }
    return out
