#!/usr/bin/env python3
"""LOI cost estimation function"""

import numpy as np
import logging
from .. import Coordinates as coord

# LOI cost function, from inertial state
def get_loi_cost_inertial(
    state_periapsis,
    radius_LLO,
    mu,
    lstar,
    tstar,
    linearApprox=False,
    message=True,
    useTargetLLO=False,
):
    """Compute LOI cost from state in inertial frame. CAREFUL WITH UNITS IN INPUT!

    Args:
        state_periapsis (np.array): cartesian state in Moon-centered inertial frame (non-dimensional)
        radius_LLO (np.array): radius of LLO, in km
        mu (float): cr3bp mass parameter (non-dimensional)
        lstar (float): length-scale, in km
        tstar (float): time-scale, in seconds
        linearApprox (bool): set to true if using linear approximation (default is False)
        message (bool): set to true if displaying message (default is True)
        useTargetLLO (bool): whether to compute LOI cost using radius_LLO or height of state at periapsis (default is False)

    Returns:
        (float): LOI cost, in km/sec

    """

    # gravitational parameter (dimensional)
    gm = mu * lstar ** 3 / tstar ** 2

    # compute parameters on the approaching trajctory
    r_peri = (
        np.sqrt(
            state_periapsis[0] ** 2 + state_periapsis[1] ** 2 + state_periapsis[2] ** 2
        )
        * lstar
    )
    v_peri = (
        np.sqrt(
            state_periapsis[3] ** 2 + state_periapsis[4] ** 2 + state_periapsis[5] ** 2
        )
        * lstar
        / tstar
    )

    # compute parameters of LLO (assume circular)
    if useTargetLLO == True:
        v_LLO = np.sqrt(gm / radius_LLO)
        c3_LLO = v_LLO ** 2 - 2 * gm / (radius_LLO)  # C3 = 2*energy [km**3/sec**2]
    else:
        v_LLO = np.sqrt(gm / r_peri)
        c3_LLO = v_LLO ** 2 - 2 * gm / (r_peri)  # C3 = 2*energy [km**3/sec**2]

    logging.debug(f"c3*: {c3_LLO}")
    logging.debug(f"v_LLO*: {v_LLO}")

    # compute c3 of transfer trajectory (c3 = 2 * energy)
    c3 = v_peri ** 2 - 2 * gm / r_peri
    logging.debug(f"c3: {c3}")

    # compute difference in c3
    delta_c3 = c3 - c3_LLO
    logging.debug(f"Delta c3: {delta_c3}")

    # compute Delta V --- fixme!
    if linearApprox == False:
        deltaV_LOI = np.sqrt(v_LLO ** 2 + delta_c3) - v_LLO

    # OR ... linear expansion-based cost
    elif linearApprox == True:
        coef1 = np.sqrt(v_LLO ** 2 - c3_LLO) - v_LLO
        logging.debug(f"Diff-term 1: {coef1}")

        coef2 = 1 / (2 * np.sqrt(v_LLO ** 2 - c3_LLO))
        logging.debug(f"Diff-term 2: {coef2}")

        deltaV_LOI = coef1 + coef2 * c3

    if message == True:
        print(f"Delta V cost: {deltaV_LOI*1000:.4f} [m/sec]")

    return deltaV_LOI


# Wrapper to get loi cost from rotating state
def get_loicost(mu, state, radius_LLO, Lstar, Tstar):
    """Wrapper to get LOI cost from state in the Earth-Moon rotating frame, centered at the Barycenter
    Args:
        mu (float):
        state (np.array): state at perilune in the Earth-Moon rotating frame, centered at the Earth-Moon barycenter
        radius_LLO (float):
        Lstar (float):
        Tstar (float):
    Returns:
        (float): loi-cost
    """
    # decompose state in rotating frame
    periluneState_MoonRot = coord.shift_state(state, -(1 - mu), axis="x")
    periluneState_MoonInrt = coord.rotating2inertial(periluneState_MoonRot, 0.0)
    # compute loi cost
    deltaV_LOI = get_loi_cost_inertial(
        periluneState_MoonInrt,
        radius_LLO,
        mu,
        Lstar,
        Tstar,
        linearApprox=False,
        message=False,
        useTargetLLO=True,
    )
    return deltaV_LOI
