#!/usr/bin/env python3
"""Propagator function wrapper in BCR4BP, planet-moon centered"""

import numpy as np
from scipy.integrate import odeint, solve_ivp

from ._rhs_functions import rhs_ccr4bp
from .output import Propout


def propagate_ccr4bp(
        state0, 
        tf, 
        mu1, 
        mu2, 
        mu3, 
        om2, 
        om3, 
        theta02, 
        theta03, 
        d2, 
        d3, 
        steps=2000, 
        stm0=False,
        events=None, 
        ivp_method="LSODA", 
        ivp_rtol=1e-12, 
        ivp_atol=1e-12, 
        state0_center="barycenter",
        force_solve_ivp=False,
        printmessg=False,
        full_output_odeint=False,
        switch_solveivp=True,
        message=False):
    """Function propagates initial cartesian state in the planet-moon centered BCR4BP using scipy's solve_ivp. 

    Args:
        mu (float): cr3bp parameter
        state0 (numpy array): numpy array containing cartesian state
        tf (float): final time of integration

        mu3 (float): scaled mass of sun
        a (float): scaled distance from sun to planet-moon barycenter
        omega_s (float): rotation rate of sun about planet-moon barycenter (should be negative)
        t0 (float): initial phase of sun
        eps (float): attenuation factor of Sun's effect (0~1)
        steps (float): number of steps to extract points (default is 2000)
        stm0 (bool): choice whether to also propagate STM (defualt is False)
        events (func): event function (default is None)
        ivp_method (str): integration method in scipy's solve_ivp() function (default is "LSODA")
        ivp_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        ivp_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)
        force_solve_ivp (bool): forcing the use of solve_ivp function (default is False)
        printmessg (bool): whether to print message from odeint (default is False)

    Returns:
        (dict): dictionary with entries:  
            (numpy array) times, xs, ys, zs, vxs, vys, vzs, stms; 
            (numpy array) statef - Cartesian state at the end of the propagation (if using event, eventState is more accurate); 
            (outputs from solve_ivp at event): eventStates, eventTimes      
    """

    if events==None and ivp_method=='LSODA' and force_solve_ivp==False:
        # use odeint
        propout, infodict = propagate_ccr4bp_odeint(state0, tf, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3, steps=steps, stm0=stm0, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol, state0_center=state0_center, full_output_odeint=True, printmessg=printmessg)
        
        # check if successful
        if infodict['message']!='Integration successful.' and switch_solveivp==True:
            if message==True:
                print('Failed with odeint(); switching to solve_ivp() for integration')
            # use solve_ivp
            propout =  propagate_ccr4bp_solve_ivp( state0, tf, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3, steps=2000, stm0=stm0, events=events, ivp_method=ivp_method, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol, state0_center=state0_center )

        # return arguments
        if full_output_odeint == False:
            return propout
        else:
            return propout, infodict

    else:
        # use solve_ivp
        infodict = {'message': 'Integration successful.'}
        propout =  propagate_ccr4bp_solve_ivp( state0, tf, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3, steps=2000, stm0=stm0, events=events, ivp_method=ivp_method, ivp_rtol=ivp_rtol, ivp_atol=ivp_atol, state0_center=state0_center )
        if full_output_odeint == False:
            return propout
        else:
            return propout, infodict


# ------------------------------------------------------------------ #
# solving with solve_ivp
def propagate_ccr4bp_solve_ivp(
        state0, 
        tf, 
        mu1, 
        mu2, 
        mu3, 
        om2, 
        om3, 
        theta02, 
        theta03, 
        d2, 
        d3, 
        steps=2000, 
        stm0=False,
        events=None, 
        ivp_method="LSODA", 
        ivp_rtol=1e-12, 
        ivp_atol=1e-12, 
        state0_center="barycenter"):
    """Function propagates initial cartesian state in the CCR4BP using scipy's solve_ivp. 
    The expected state is centered at the barycenter, and is shifted to m1!

    Note:
        For EoM in m1-m2 rotating frame, set theta02 = 0 and om2 = 1 and d2 = 1

    Args:
        state0 (numpy array): numpy array containing cartesian state, centered at barycenter or first primary (toggle state0_center)
        tf (float): final time of integration
        mu1 (float): scaled mass parameter of first primary
        mu2 (float): scaled mass parameter of second primary
        mu3 (float): scaled mass parameter of third primary
        om2 (float): angular rotation rate of m1-m2 system, in radians/(canonical time)
        om3 (float): angular rotation rate of m1-m2 system, in radians/(canonical time)
        theta02 (float): initial angle between x-line and third primary, in radians
        theta03 (float): initial angle between x-line and third primary, in radians
        d2 (float): distance from first to second primary
        d3 (float): distance from first to third primary
        steps (float): number of steps to extract points (default is 2000)
        stm0 (bool): choice whether to also propagate STM (defualt is False)
        events (func): event function (default is None)
        ivp_method (str): integration method in scipy's solve_ivp() function (default is "LSODA")
        ivp_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        ivp_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)
        state0_center (str): if "barycenter", expect input state0 to be centered at the barycenter, and the output is also centered at the barycenter. Else, both are centered at m1. 

    Returns:
        (dict): dictionary with entries:  
            (numpy array) times, xs, ys, zs, vxs, vys, vzs, stms; 
            (numpy array) statef - Cartesian state at the end of the propagation (if using event, eventState is more accurate); 
            (outputs from solve_ivp at event): eventStates, eventTimes      
    """

    if state0_center=="barycenter":
        state0cc = state0
        state0cc[0] = state0[0] + mu2
    else:
        state0cc = state0

    timesay = np.linspace(0, tf, steps)
    sol = solve_ivp(fun=rhs_ccr4bp, t_span=(0,tf), y0=state0cc, events=events, t_eval=timesay, args=(mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3), method=ivp_method, rtol=ivp_rtol, atol=ivp_atol)
    # unpack result
    times = sol.t       # time
    if state0_center=="barycenter":
        xs  = sol.y[0] - mu2   # shift back
    else:
        xs  = sol.y[0]
    ys  = sol.y[1]
    zs  = sol.y[2]
    vxs = sol.y[3]
    vys = sol.y[4]
    vzs = sol.y[5]
    # stms is not returned (just a place-holder)
    stms = np.zeros((1,))
    # create numpy array of state at final time of propagation
    statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
    # evaluate rhs based on final state
    try:
        dstatef = rhs_ccr4bp(times[-1], statef, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3)
    except:
        dstatef = []
    # check for events
    if events is None:
        # no assignment if no event function is provided
        eventStates = []
        eventTimes = []
    else:
        # return event state and time
        eventStates = sol.y_events
        eventTimes = sol.t_events
    
    # prepare output dictionary
    out = {"times":times, "xs":xs, "ys":ys, "zs":zs, "vxs":vxs, "vys":vys, "vzs":vzs, "state0": state0cc,
            "statef":statef, "dstatef": dstatef, "stms":stms, "eventStates":eventStates, "eventTimes":eventTimes}

    return out



    
# ------------------------------------------------------------------ #
# solving with odeint()
def propagate_ccr4bp_odeint(
        state0, 
        tf, 
        mu1, 
        mu2, 
        mu3, 
        om2, 
        om3, 
        theta02, 
        theta03, 
        d2, 
        d3, 
        steps=2000, 
        stm0=False,
        ivp_rtol=1e-12, 
        ivp_atol=1e-12, 
        state0_center="barycenter",
        full_output_odeint=False,
        printmessg=False):
    """Function propagates initial cartesian state in CCR4BP using scipy's odeint. 

    Args:
        state0 (numpy array): numpy array containing cartesian state, centered at barycenter or first primary (toggle state0_center)
        tf (float): final time of integration
        mu1 (float): scaled mass parameter of first primary
        mu2 (float): scaled mass parameter of second primary
        mu3 (float): scaled mass parameter of third primary
        om2 (float): angular rotation rate of m1-m2 system, in radians/(canonical time)
        om3 (float): angular rotation rate of m1-m2 system, in radians/(canonical time)
        theta02 (float): initial angle between x-line and third primary, in radians
        theta03 (float): initial angle between x-line and third primary, in radians
        d2 (float): distance from first to second primary
        d3 (float): distance from first to third primary
        steps (float): number of steps to extract points (default is 2000)
        stm0 (bool): choice whether to also propagate STM (defualt is False)
        events (func): event function (default is None)
        ivp_method (str): integration method in scipy's solve_ivp() function (default is "LSODA")
        ivp_rtol (float): relative tolerance in solve_ivp() function (default is 1e-12)
        ivp_atol (float): absolute tolerance in solve_ivp() function (default is 1e-12)
        state0_center (str): if "barycenter", expect input state0 to be centered at the barycenter, and the output is also centered at the barycenter. Else, both are centered at m1. 

    Returns:
        (dict): dictionary with entries:  
            (numpy array) times, xs, ys, zs, vxs, vys, vzs, stms; 
            (numpy array) statef - Cartesian state at the end of the propagation (if using event, eventState is more accurate);     
    """

    if state0_center=="barycenter":
        state0cc = state0
        state0cc[0] = state0[0] + mu2
    else:
        state0cc = state0

    # construct time-array where state will be returned
    timesay = np.linspace(0, tf, steps)
    
    # if no STM is provided, only propagate the Cartesian state (i.e. integrate 6 differential equations)
    if stm0==False:    
        
        # propagate state
        if full_output_odeint==False:
            sol = odeint(func=rhs_ccr4bp, y0=state0, t=timesay, args=(mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)
        else:
            sol, optout = odeint(func=rhs_ccr4bp, y0=state0, t=timesay, args=(mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)

        # unpack cartesian state and time
        times  = timesay       # time
        xs  = sol[:,0]
        ys  = sol[:,1]
        zs  = sol[:,2]
        vxs = sol[:,3]
        vys = sol[:,4]
        vzs = sol[:,5]
        # stms is not returned (just a place-holder)
        stms = np.zeros((1,))
         # create numpy array of state at final time of propagation
        statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
        # evaluate rhs based on final state
        try:
          dstatef = rhs_ccr4bp(times[-1], statef, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3)
        except:
            dstatef = []

    # if initial STM is provided, also propagate STM (i.e. integrate 6+36=42 differential equations)
    else:
       raise Exception("Not implemented!")
      #   # extend state to include STM
      #   state0ext = np.zeros((42,))
      #   # store cartesian state
      #   state0ext[:6] = state0
      #   # store initial identity stm into extended state vector
      #   state0ext[5+1]  = 1
      #   state0ext[5+8]  = 1
      #   state0ext[5+15] = 1
      #   state0ext[5+22] = 1
      #   state0ext[5+29] = 1
      #   state0ext[5+36] = 1
      #   # propagate state and stm
      #   if full_output_odeint == False:
      #       if jitoption==True:
      #           sol = odeint(func=rhs_bcr4bp_planetmoon_with_STM_numba, y0=state0ext, t=timesay, args=(mu,mu3,a,t0,omega_s,eps), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)
      #       else:
      #           sol = odeint(func=rhs_bcr4bp_planetmoon_with_STM, y0=state0ext, t=timesay, args=(mu,mu3,a,t0,omega_s,eps), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)
      #   else:
      #       if jitoption==True:
      #           sol, optout = odeint(func=rhs_bcr4bp_planetmoon_with_STM_numba, y0=state0ext, t=timesay, args=(mu,mu3,a,t0,omega_s,eps), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)
      #       else:
      #           sol, optout = odeint(func=rhs_bcr4bp_planetmoon_with_STM, y0=state0ext, t=timesay, args=(mu,mu3,a,t0,omega_s,eps), Dfun=None, col_deriv=0, full_output=full_output_odeint, ml=None, mu=None, rtol=ivp_rtol, atol=ivp_atol, tcrit=None, h0=0.0, hmax=0.0, hmin=0.0, ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=printmessg, tfirst=True)

      #   # unpack cartesian state and time
      #   times  = timesay       # time
      #   xs  = sol[:,0]
      #   ys  = sol[:,1]
      #   zs  = sol[:,2]
      #   vxs = sol[:,3]
      #   vys = sol[:,4]
      #   vzs = sol[:,5]
      #   # unpack STM
      #   stms = sol[:,6:].T
      #   # create numpy array of state at final time of propagation
      #   statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
      #   # evaluate rhs based on final state
      #   dstatef = rhs_bcr4bp_planetmoon(times[-1], statef, mu, mu3, a, t0, omega_s, eps)

    # create numpy array of state at final time of propagation
    statef = np.array([xs[-1], ys[-1], zs[-1], vxs[-1], vys[-1], vzs[-1]])
    
    # prepare output dictionary
    out = {"times":times, "xs":xs, "ys":ys, "zs":zs, "vxs":vxs, "vys":vys, "vzs":vzs, "state0": state0cc,
            "statef":statef, "dstatef": dstatef, "stms":stms}

    if full_output_odeint==False:
        return out
    else:
        return out, optout