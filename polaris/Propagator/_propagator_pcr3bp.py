#!/usr/bin/env python3
"""
Propagator functions for cr3bp
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint, solve_ivp
from numba import jit

from ._rhs_functions import rhs_pcr3bp, rhs_pcr3bp_with_STM


# ---------------------------------------------------------------------------------------- #
def propagate_pcr3bp(mu, state0, tf, steps=200, t0=0.0, stm_option=False, events=None, ivp_method='LSODA', ivp_rtol=1e-12, ivp_atol=1e-12, force_solve_ivp=False, switch2solveivp=True, message=False):
    """Propagator function for CR3BP, either in 2D or 3D. 
    The function calls either scipy.integrate.odeint() or scipy.integrate.solve_ivp()
    odeint() is used if method is 'LSODA' and events=None or force_solve_ivp=False

    Args:
        mu (float): cr3bp parameter
        state0 (numpy array): numpy array containing cartesian state, length 4 for planar case
        tf (float): final time of integration
        steps (float): number of steps to extract points (default is 200)
        t0 (float): initial time
        stm_option (bool): choice whether to also propagate STM (defualt is False)
        events (func): event function (default is None)
        ivp_method (str): integration method in scipy's solve_ivp() function (default is "LSODA")
        ivp_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        ivp_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)
        force_solve_ivp (bool): forcing the use of solve_ivp function (default is False)
        switch2solveivp (bool): if set to True, when integration is unsuccessful with odeint, function is switched to solve_ivp()
        message (bool): whether to display message when switching from odeint() to solve_ivp(), default is False
    Returns:
        (dict): dictionary with entries "xs", "ys", "zs", "vxs", "vys", "vzs", "times", "stms", "statef", "dstatef", "eventStates", "eventTimes"
    """
    # decides whether to use solve_ivp or odeint
    if events==None and ivp_method=='LSODA' and force_solve_ivp==False:
        # use odeint
        propout, infodict = propagate_pcr3bp_odeint(mu, state0, tf, steps=steps, t0=t0, stm_option=stm_option, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol)
        # check if integration is successfully done
        if infodict['message']!='Integration successful.' and switch2solveivp==True:
            if message==True:
                print('Failed with odeint(); switching to solve_ivp() for integration')
            propout = propagate_pcr3bp_solve_ivp(mu, state0, tf, steps=steps, t0=t0, stm_option=stm_option, events=events, ivp_method=ivp_method, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol)
    else:
        # use solve_ivp
        propout = propagate_pcr3bp_solve_ivp(mu, state0, tf, steps=steps, t0=t0, stm_option=stm_option, events=events, ivp_method=ivp_method, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol)
    return propout


# ---------------------------------------------------------------------------------------- #
def propagate_pcr3bp_odeint(mu, state0, tf, steps=2000, t0=0.0, stm_option=False, ivp_rtol=1e-12, ivp_atol=1e-12):
    """Propagator for CR3BP using odeint()"""
    # construct time-array where state will be returned
    time_array = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option==False:

        # 2D propagation
        sol, infodict = odeint(func=rhs_pcr3bp, y0=state0, t=time_array, args=(mu,), Dfun=None, col_deriv=0, full_output=1, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=True)
        # unpack cartesian state and time
        times  = time_array       # time
        x_arr  = sol[:,0]
        y_arr  = sol[:,1]
        vx_arr = sol[:,2]
        vy_arr = sol[:,3]
        # stmmat is not returned (just a place-holder)
        stmmat = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        # extend state to include STM
        state0ext = np.zeros((42,))    # FIXME! (ugly as of now)
        state0_modified = np.array([state0[0], state0[1], 0.0, state0[2], state0[3], 0.0])
        # store cartesian state
        state0ext[:6] = state0_modified
        # store initial identity stm into extended state vector
        state0ext[5+1]  = 1
        state0ext[5+8]  = 1
        state0ext[5+15] = 1
        state0ext[5+22] = 1
        state0ext[5+29] = 1
        state0ext[5+36] = 1
        # propagate state and stm
        sol, infodict = odeint(func=rhs_pcr3bp_with_STM, y0=state0ext, t=time_array, args=(mu,), Dfun=None, col_deriv=0, full_output=1, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=True)
        # unpack cartesian state and time
        times = time_array       # time
        x_arr  = sol[:,0]
        y_arr  = sol[:,1]
        vx_arr = sol[:,3]
        vy_arr = sol[:,4]
        # unpack STM
        stmmat = sol[:,6:].T   # MUST FIX!

    # create numpy array of state at final time of propagation
    statef = np.array([x_arr[-1], y_arr[-1], vx_arr[-1], vy_arr[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_pcr3bp(times[-1], statef, mu)
    # prepare output dictionary
    out = {
        "times": times,
        "xs": x_arr,
        "ys": y_arr,
        "vxs": vx_arr,
        "vys": vy_arr,
        "state0": state0,
        "statef": statef,
        "stms":stmmat,
        "dstatef":dstatef,
        "eventStates": [],
        "eventTimes": []
    }
    return out, infodict



# ---------------------------------------------------------------------------------------- #
def propagate_pcr3bp_solve_ivp(mu, state0, tf, steps=2000,t0=0.0, stm_option=False, events=None, ivp_method="LSODA", ivp_rtol=1e-12, ivp_atol=1e-12):
    """Propagator for CR3BP using solve_ivp()"""
    # construct time-array where state will be returned
    time_array = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option==False:
        # propagate state
        sol = solve_ivp(fun=rhs_pcr3bp, t_span=(0,tf), y0=state0, events=events, t_eval=time_array, args=(mu,), method=ivp_method, rtol=ivp_rtol, atol=ivp_atol)
        # unpack cartesian state and time
        times = sol.t       # time
        x_arr  = sol.y[0]
        y_arr  = sol.y[1]
        vx_arr = sol.y[3]
        vy_arr = sol.y[4]
        # stmmat is not returned (just a place-holder)
        stmmat = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        # planar propagation # TODO: need pure impementation...
        # extend state to include STM
        state0ext = np.zeros((42,))
        state0_modified = np.array([state0[0], state0[1], 0.0, state0[2], state0[3], 0.0])
        # store cartesian state
        state0ext[:6] = state0_modified
        # store initial identity stm into extended state vector
        state0ext[5+1]  = 1
        state0ext[5+8]  = 1
        state0ext[5+15] = 1
        state0ext[5+22] = 1
        state0ext[5+29] = 1
        state0ext[5+36] = 1
        # propagate state and stm
        sol = solve_ivp(fun=rhs_pcr3bp_with_STM, t_span=(0,tf), y0=state0ext, events=events, t_eval=time_array, args=(mu,), method=ivp_method, rtol=ivp_rtol, atol=ivp_atol)
        # unpack cartesian state an#d time
        times = sol.t       # time
        x_arr  = sol.y[0]
        y_arr  = sol.y[1]
        vx_arr = sol.y[3]
        vy_arr = sol.y[4]
        # unpack STM
        stmmat = sol.y[6:,:]

    # return events
    if events is None:
        eventStates = []
        eventTimes = []
    else:
        eventStates = sol.y_events
        eventTimes = sol.t_events

    # create numpy array of state at final time of propagation
    statef = np.array([x_arr[-1], y_arr[-1], vx_arr[-1], vy_arr[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_pcr3bp(times[-1], statef, mu)
    # prepare output dictionary
    out = {
        "times": times,
        "xs": x_arr,
        "ys": y_arr,
        "vxs": vx_arr,
        "vys": vy_arr,
        "state0": state0,
        "statef": statef,
        "stms":stmmat,
        "dstatef":dstatef,
        "eventStates": [],
        "eventTimes": []
    }
    return out
