#!/usr/bin/env python3
"""
Propagator functions for two-body
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint, solve_ivp
from numba import jit

from ._rhs_functions import rhs_twobody_j2
from .output import Propout


# ---------------------------------------------------------------------------------------- #
def propagate_twobody_j2(
    mu,
    j2,
    re,
    state0,
    tf,
    steps=2000,
    t0=0.0,
    stm_option=False,
    events=None,
    ivp_method="LSODA",
    ivp_rtol=1e-12,
    ivp_atol=1e-12,
    force_solve_ivp=False,
):
    """Propagator function for two-body.
    The function calls either scipy.integrate.odeint() or scipy.integrate.solve_ivp()
    odeint() is used if method is 'LSODA' and events=None or force_solve_ivp=False

    Args:
        mu (float): cr3bp parameter
        state0 (numpy array): numpy array containing cartesian state
        tf (float): final time of integration
        steps (float): number of steps to extract points (default is 2000)
        t0 (float): initial time
        stm_option (bool): choice whether to also propagate STM (defualt is False)
        events (func): event function (default is None)
        ivp_method (str): integration method in scipy's solve_ivp() function (default is "LSODA")
        ivp_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        ivp_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)
        force_solve_ivp (bool): forcing the use of solve_ivp function (default is False)

    Returns:
        (dict): dictionary with entries "xs", "ys", "zs", "vxs", "vys", "vzs", "times", "stms", "statef", "dstatef", "eventStates", "eventTimes"
    """
    # decides whether to use solve_ivp or odeint
    if events == None and ivp_method == "LSODA" and force_solve_ivp == False:
        # use odeint
        propout = propagate_twobody_odeint_j2(
            mu,
            j2,
            re,
            state0,
            tf,
            steps=steps,
            t0=t0,
            stm_option=stm_option,
            ivp_rtol=ivp_rtol,
            ivp_atol=ivp_atol,
        )
    else:
        # use solve_ivp
        propout = propagate_twobody_solve_ivp_j2(
            mu,
            j2,
            re,
            state0,
            tf,
            steps=steps,
            t0=t0,
            stm_option=stm_option,
            events=events,
            ivp_method=ivp_method,
            ivp_rtol=ivp_rtol,
            ivp_atol=ivp_atol,
        )
    return propout


# ---------------------------------------------------------------------------------------- #
def propagate_twobody_odeint_j2(
    mu,
    j2,
    re,
    state0,
    tf,
    steps=2000,
    t0=0.0,
    stm_option=False,
    ivp_rtol=1e-12,
    ivp_atol=1e-12,
):
    """Propagator for two-body using odeint()"""
    # construct time-array where state will be returned
    timesay = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option == False:
        # propagate state
        sol = odeint(
            func=rhs_twobody_j2,
            y0=state0,
            t=timesay,
            args=(
                mu,
                j2,
                re,
            ),
            Dfun=None,
            col_deriv=0,
            full_output=0,
            ml=None,
            mu=None,
            rtol=ivp_rtol,
            atol=ivp_atol,
            tcrit=None,
            h0=0.0,
            hmax=0.0,
            hmin=0.0,
            ixpr=0,
            mxstep=0,
            mxhnil=0,
            mxordn=12,
            mxords=5,
            printmessg=0,
            tfirst=True,
        )

        # unpack cartesian state and time
        times = timesay  # time
        xs = sol[:, 0]
        ys = sol[:, 1]
        zs = sol[:, 2]
        vxs = sol[:, 3]
        vys = sol[:, 4]
        vzs = sol[:, 5]
        # stms is not returned (just a place-holder)
        stms = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        raise NotImplementedError

    # create numpy array of state at final time of propagation
    statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_twobody_j2(times[-1], statef, mu, j2, re)

    # prepare output
    return Propout(
        xs, ys, zs, vxs, vys, vzs, times, state0, statef, stms, dstatef, None, None
    )


# ---------------------------------------------------------------------------------------- #
def propagate_twobody_solve_ivp_j2(
    mu,
    j2,
    re,
    state0,
    tf,
    steps=2000,
    t0=0.0,
    stm_option=False,
    events=None,
    ivp_method="LSODA",
    ivp_rtol=1e-12,
    ivp_atol=1e-12,
):
    """Propagator for two-body using solve_ivp()"""
    # construct time-array where state will be returned
    timesay = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option == False:
        # propagate state
        sol = solve_ivp(
            fun=rhs_twobody_j2,
            t_span=(0, tf),
            y0=state0,
            events=events,
            t_eval=timesay,
            args=(
                mu,
                j2,
                re,
            ),
            method=ivp_method,
            rtol=ivp_rtol,
            atol=ivp_atol,
        )
        # unpack cartesian state and time
        times = sol.t  # time
        xs = sol.y[0]
        ys = sol.y[1]
        zs = sol.y[2]
        vxs = sol.y[3]
        vys = sol.y[4]
        vzs = sol.y[5]
        # stms is not returned (just a place-holder)
        stms = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        raise NotImplementedError

    # create numpy array of state at final time of propagation
    statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_twobody_j2(times[-1], statef, mu, j2, re)

    # return events
    if events is None:
        eventStates = None
        eventTimes = None
    else:
        eventStates = sol.y_events
        eventTimes = sol.t_events

    # prepare output
    return Propout(
        xs,
        ys,
        zs,
        vxs,
        vys,
        vzs,
        times,
        state0,
        statef,
        stms,
        dstatef,
        eventStates,
        eventTimes,
    )
