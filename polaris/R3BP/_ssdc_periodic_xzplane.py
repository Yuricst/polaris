#!/usr/bin/env python3
"""
Single shooting differential correction for perpendicular trajectories about xz-plane
"""


import numpy as np
import numpy.linalg as la
from numba import jit


from ..Propagator import propagate_cr3bp, rhs_cr3bp, ssdc


# function for single shooting differential correction of periodic trajectories about xz-plane
def ssdc_periodic_xzplane(
    cr3bp_param, state0, period0, fix="period", tolDC=1e-11, iter_max=10, message=False
):
    """Wrapper for single-shooting differential correction for periodic trajectory with symmetry about xz-plane
    Args:
        cr3bp_param (R3BP.Parameters): CR3BP system parameters
        state0 (np.array): initial guess state-vector, in canonical units
        period0 (float): initial guess period, in canonical units
        fix (str): parameter to keep fixed during differential correction, "period" or "z"
        tolDC (float): tolerance on differential correction
        iter_max (int): maximum allowable iteration
        message (bool): whether to display message during the process
    Returns:
        (tuple): tuple of period, state, and convergence flag (1 or 0)
    """
    if np.abs(state0[1]) > 1e-9:
        print("WARNING: initial state may not lie on xy-plane")

    # initialize
    state, thalf = state0, period0 / 2
    conv_flag = 0
    # iterate over counter
    for iteration in range(iter_max):
        # propagate half-way
        propout = propagate_cr3bp(
            cr3bp_param,
            state,
            thalf,
            stm_option=True,
            ivp_method="LSODA",
            ivp_rtol=1e-12,
            ivp_atol=1e-12,
        )
        if fix == "z":
            xi = np.array([state[0], state[4], thalf])
        elif fix == "period":
            xi = np.array([state[0], state[2], state[4]])
        # compute error
        ferr = np.array(
            [propout.statef[1], propout.statef[3], propout.statef[5]]
        )
        # state-transition matrix
        stm = np.reshape(propout.stms[:, -1], (6, 6))
        # compute Jacobian
        if fix == "z":  # fix_period==False and fix_amplitude==True:
            df = np.array(
                [
                    [stm[1, 0], stm[1, 4], propout.dstatef[1]],
                    [stm[3, 0], stm[3, 4], propout.dstatef[3]],
                    [stm[5, 0], stm[5, 4], propout.dstatef[5]],
                ]
            )
        elif fix == "period":  # fix_period==True and fix_amplitude==False:
            df = np.array(
                [
                    [stm[1, 0], stm[1, 2], stm[1, 4]],
                    [stm[3, 0], stm[3, 2], stm[3, 4]],
                    [stm[5, 0], stm[5, 2], stm[5, 4]],
                ]
            )
        # apply ssdc
        xii = ssdc(xi, ferr, df)

        # update state from free variable vector
        if fix == "z":
            state[0], state[4], thalf = xii[0], xii[1], xii[2]
        elif fix == "period":
            state[0], state[2], state[4] = xii[0], xii[1], xii[2]

        if la.norm(ferr) < tolDC:
            if message == True:
                print(
                    f"Cleared tolerance at iter {iteration+1}  with error {la.norm(ferr)}"
                )
            conv_flag = 1
            break

        if message == True:
            print(f"*** Current error at iter {iteration+1}: {la.norm(ferr)} ***")

        # update control vector
        xi = xii

    # return result
    return thalf * 2, state, conv_flag
