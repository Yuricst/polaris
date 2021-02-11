#!/usr/bin/env python3
"""
Propagator functions for two-body
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import odeint, solve_ivp
from numba import jit

from ._rhs_functions import rhs_twobody


# ---------------------------------------------------------------------------------------- #
def propagate_twobody(mu, state0, tf, steps=2000, t0=0.0, stm_option=False, events=None, ivp_method='LSODA', ivp_rtol=1e-12, ivp_atol=1e-12, force_solve_ivp=False):
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
    if events==None and ivp_method=='LSODA' and force_solve_ivp==False:
        # use odeint
        propout = propagate_twobody_odeint(mu, state0, tf, steps=steps, t0=t0, stm_option=stm_option, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol)
    else:
        # use solve_ivp
        propout = propagate_twobody_solve_ivp(mu, state0, tf, steps=steps,t0=t0, stm_option=stm_option, events=events, ivp_method=ivp_method, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol)
    return propout


# ---------------------------------------------------------------------------------------- #
def propagate_twobody_odeint(mu, state0, tf, steps=2000, t0=0.0, stm_option=False, ivp_rtol=1e-12, ivp_atol=1e-12):
    """Propagator for two-body using odeint()"""
    # construct time-array where state will be returned
    time_array = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option==False:    
        # propagate state
        sol = odeint(func=rhs_twobody, y0=state0, t=time_array, args=(mu,), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=True)
        
        # unpack cartesian state and time
        times  = time_array       # time
        x_arr  = sol[:,0]
        y_arr  = sol[:,1]
        z_arr  = sol[:,2]
        vx_arr = sol[:,3]
        vy_arr = sol[:,4]
        vz_arr = sol[:,5]
        # stmmat is not returned (just a place-holder)
        stmmat = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        raise Exception("Not implemented yet!")
        # # extend state to include STM
        # state0ext = np.zeros((42,))
        # # store cartesian state
        # state0ext[:6] = state0
        # # store initial identity stm into extended state vector
        # state0ext[5+1]  = 1
        # state0ext[5+8]  = 1
        # state0ext[5+15] = 1
        # state0ext[5+22] = 1
        # state0ext[5+29] = 1
        # state0ext[5+36] = 1
        # # propagate state and stm
        # sol = odeint(func=rhs_cr3bp_with_STM_numba, y0=state0ext, t=time_array, args=(mu,), Dfun=None, col_deriv=0, full_output=0, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0, tfirst=True)

        # # unpack cartesian state and time
        # times = time_array       # time
        # x_arr  = sol[:,0]
        # y_arr  = sol[:,1]
        # z_arr  = sol[:,2]
        # vx_arr = sol[:,3]
        # vy_arr = sol[:,4]
        # vz_arr = sol[:,5]

        # # unpack STM
        # stmmat = sol[:,6:].T
    
    # create numpy array of state at final time of propagation
    statef = np.array([x_arr[-1], y_arr[-1], z_arr[-1], vx_arr[-1], vy_arr[-1], vz_arr[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_twobody(times[-1], statef, mu)

    # prepare output dictionary
    out = {
        "times": times,
        "xs": x_arr,
        "ys": y_arr,
        "zs": z_arr,
        "vxs": vx_arr,
        "vys": vy_arr,
        "vzs": vz_arr,
        "state0": state0,
        "statef": statef, 
        "stms":stmmat, 
        "dstatef":dstatef, 
        "eventStates": [], 
        "eventTimes": []
    }
    return out


# ---------------------------------------------------------------------------------------- #
def propagate_twobody_solve_ivp(mu, state0, tf, steps=2000,t0=0.0, stm_option=False, events=None, ivp_method="LSODA", ivp_rtol=1e-12, ivp_atol=1e-12):
    """Propagator for two-body using solve_ivp()"""
    # construct time-array where state will be returned
    time_array = np.linspace(t0, tf, steps)

    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm_option==False:    
        # propagate state
        sol = solve_ivp(fun=rhs_twobody, t_span=(0,tf), y0=state0, events=events, t_eval=time_array, args=(mu,), method=ivp_method, rtol=ivp_rtol, atol=ivp_atol)
        # unpack cartesian state and time
        times  = sol.t       # time
        x_arr  = sol.y[0]
        y_arr  = sol.y[1]
        z_arr  = sol.y[2]
        vx_arr = sol.y[3]
        vy_arr = sol.y[4]
        vz_arr = sol.y[5]
        # stmmat is not returned (just a place-holder)
        stmmat = np.zeros((1,))

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
        raise Exception("Not implemented yet!")
        # extend state to include STM
        # state0ext = np.zeros((42,))
        # # store cartesian state
        # state0ext[:6] = state0
        # # store initial identity stm into extended state vector
        # state0ext[5+1]  = 1
        # state0ext[5+8]  = 1
        # state0ext[5+15] = 1
        # state0ext[5+22] = 1
        # state0ext[5+29] = 1
        # state0ext[5+36] = 1
        # # propagate state and stm
        # sol = solve_ivp(fun=rhs_twobody, t_span=(0,tf), y0=state0ext, events=events, t_eval=time_array, args=(mu,), method=ivp_method, rtol=ivp_rtol, atol=ivp_atol)
        # # unpack cartesian state an#d time
        # times = sol.t       # time
        # x_arr  = sol.y[0]
        # y_arr  = sol.y[1]
        # z_arr  = sol.y[2]
        # vx_arr = sol.y[3]
        # vy_arr = sol.y[4]
        # vz_arr = sol.y[5]
        # # unpack STM
        # stmmat = sol.y[6:,:]
    
    # create numpy array of state at final time of propagation
    statef = np.array([x_arr[-1], y_arr[-1], z_arr[-1], vx_arr[-1], vy_arr[-1], vz_arr[-1]])
    # evaluate rhs based on final state
    dstatef = rhs_twobody(times[-1], statef, mu)

    # return events
    if events is None:
        eventStates = []
        eventTimes = []
    else:
        eventStates = sol.y_events
        eventTimes = sol.t_events

    # prepare output dictionary
    out = {
        "times": times,
        "xs": x_arr,
        "ys": y_arr,
        "zs": z_arr,
        "vxs": vx_arr,
        "vys": vy_arr,
        "vzs": vz_arr,
        "state0": state0,
        "statef": statef, 
        "stmmat":stmmat, 
        "dstatef":dstatef, 
        "eventStates":eventStates, 
        "eventTimes":eventTimes
    }
    return out


