#!/usr/bin/env python3
"""
jitcode integrator for CR3BP and PCR3BP
Yuri Shimane, 2021/01/05
"""

import numpy as np
from jitcode import jitcode
from jitcode import y as ystate



# ---------------------------------------------------------------------------------------------------------------------------------------- #
# CR3BP integrator
def propagate_cr3bp_jitcode(mu, state0, tf, steps=2000, ODEobject=None, jitcode_method='lsoda', jitcode_rtol=1e-12, jitcode_atol=1e-12):
    """Function propagates initial cartesian state in the CR3BP using JiTCODE. If the function has been called once already, and the RHS parameters (mu) is remains the same, then the ODE object should be passed as ODEobject. 
    
    Args:
        mu (float): cr3bp parameter
        state0 (numpy array): numpy array containing cartesian states x, y, z, vx, vy, vz
        tf (float): final time of integration
        steps (float): number of steps to extract points (default is 2000)
        ODEobject (ODE object): if there was previously created ODE object, pass it; else set to None
        stm0 (bool): choice whether to also propagate STM (defualt is False)
        jitcode_method (str): integration method in scipy's solve_ivp() function (default is "lsoda")
        jitcode_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        jitcode_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)

    Returns:
        (tuple): tuple of export dictionary, and ODE object
            dictionary with entries:  
                (numpy array) times, xs, ys, zs, vxs, vys, vzs 
                (numpy array) statef - Cartesian state at the end of the propagation (if using event, eventState is more accurate)   
    """

    # initialize ODE object
    if ODEobject==None:
        # right-hand side
        f_cr3bp = [
            # position derivatives
            ystate(3) ,
            ystate(4) ,
            ystate(5) ,
            # velocitystate derivatives
            2*ystate(4) + ystate(0) - ((1-mu)/((ystate(0)+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*(mu+ystate(0)) + (mu/ ((ystate(0)-1+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*(1-mu-ystate(0)) ,
            -2*ystate(3) + ystate(1) - ((1-mu)/((ystate(0)+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*ystate(1) - (mu/ ((ystate(0)-1+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*ystate(1) ,
            -((1-mu)/((ystate(0)+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*ystate(2) - (mu/ ((ystate(0)-1+mu)**2 + ystate(1)**2 + ystate(2)**2)**1.5)*ystate(2)
        ]

        ODE = jitcode(f_cr3bp)
        ODE.set_integrator(name=jitcode_method, rtol=jitcode_rtol, atol=jitcode_atol)
    else:
        ODE = ODEobject

    # initialize storage for time and state
    ODE.set_initial_value(state0, time=0.0)
    times = np.linspace(0, tf, steps)
    datanp = np.zeros((len(times), 6))
    for i in range(len(times)):
        datanp[i, :] = ODE.integrate(times[i])
    # export solution
    statef = np.array([ datanp[-1,0], datanp[-1,1], datanp[-1,2], datanp[-1,3], datanp[-1,4], datanp[-1,5] ])
    soldict = {"times": times, "xs": datanp[:,0], "ys": datanp[:,1], "zs": datanp[:,2], "vxs": datanp[:,3], "vys": datanp[:,4], "vzs": datanp[:,5], "statef": statef }

    return soldict, ODE



# ---------------------------------------------------------------------------------------------------------------------------------------- #
# PCR3BP integrator
def propagate_pcr3bp_jitcode(mu, state0, tf, steps=2000, ODEobject=None, jitcode_method='lsoda', jitcode_rtol=1e-12, jitcode_atol=1e-12):
    """Function propagates initial cartesian state in the PCR3BP using JiTCODE. If the function has been called once already, and the RHS parameters (mu) is remains the same, then the ODE object should be passed as ODEobject. 
    
    Args:
        mu (float): cr3bp parameter
        state0 (numpy array): numpy array containing cartesian states x, y, vx, vy
        tf (float): final time of integration
        steps (float): number of steps to extract points (default is 2000)
        ODEobject (ODE object): if there was previously created ODE object, pass it; else set to None
        stm0 (bool): choice whether to also propagate STM (defualt is False)
        jitcode_method (str): integration method in scipy's solve_ivp() function (default is "lsoda")
        jitcode_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        jitcode_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)

    Returns:
        (tuple): tuple of export dictionary, and ODE object
            dictionary with entries:  
                (numpy array) times, xs, ys, zs, vxs, vys, vzs 
                (numpy array) statef - Cartesian state at the end of the propagation (if using event, eventState is more accurate)   
    """

    # initialize ODE object
    if ODEobject==None:
        # right-hand side
        f_pcr3bp = [
            # position derivatives
            ystate(2) ,   # (3) -> (2)
            ystate(3) ,   # (4) -> (3)
            # velocity derivatives
            2*ystate(3) + ystate(0) - ((1-mu)/((ystate(0)+mu)**2 + ystate(1)**2)**1.5)*(mu+ystate(0)) + (mu/ ((ystate(0)-1+mu)**2 + ystate(1)**2)**1.5)*(1-mu-ystate(0)) ,
            -2*ystate(2) + ystate(1) - ((1-mu)/((ystate(0)+mu)**2 + ystate(1)**2)**1.5)*ystate(1) - (mu/ ((ystate(0)-1+mu)**2 + ystate(1)**2)**1.5)*ystate(1) 
        ]

        ODE = jitcode(f_pcr3bp)
        ODE.set_integrator(name=jitcode_method, rtol=jitcode_rtol, atol=jitcode_atol)
    else:
        ODE = ODEobject

    # initialize storage for time and state
    ODE.set_initial_value(state0, time=0.0)
    times = np.linspace(0, tf, steps)
    datanp = np.zeros((len(times), 4))
    for i in range(len(times)):
        datanp[i, :] = ODE.integrate(times[i])
    # export solution
    statef = np.array([ datanp[-1,0], datanp[-1,1], datanp[-1,2], datanp[-1,3] ])
    soldict = {"times": times, "xs": datanp[:,0], "ys": datanp[:,1], "vxs": datanp[:,2], "vys": datanp[:,3], "statef": statef }

    return soldict, ODE


