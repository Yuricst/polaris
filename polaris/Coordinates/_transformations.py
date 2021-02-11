#!/usr/bin/env python3
"""
Functions for transforming states
"""


import numpy as np
from numba import jit
import numpy.linalg as la


# rotational matrices
@jit(nopython=True)
def rotmat_ax1(theta):
    """Rotation matrix about first axis"""
    return np.array([ [ 1.0, 0.0,            0.0 ], 
                      [ 0.0, np.cos(theta), -np.sin(theta) ], 
                      [ 0.0, np.sin(theta),  np.cos(theta) ] ])

@jit(nopython=True)
def rotmat_ax2(theta):
    """Rotation matrix about first axis"""
    return np.array([ [  np.cos(theta), 0.0, np.sin(theta) ], 
                      [  0.0,           1.0, 0.0           ], 
                      [ -np.sin(theta), 0.0, np.cos(theta) ] ])

@jit(nopython=True)
def rotmat_ax3(theta):
    """Rotation matrix about third axis"""
    return np.array([ [ np.cos(theta), -np.sin(theta), 0.0 ],
                      [ np.sin(theta),  np.cos(theta), 0.0 ], 
                      [ 0.0,            0.0,           1.0 ] ])


# convert state from dimensional to canonical
@jit(nopython=True)
def dim2nondim(state, lstar, tstar):
    """Convert from dimensional to non-dimensional state
    
    Args:
        state (np.array): length 6 array of Cartesian state, in units km, km/sec
        lstar (np.array): characteristic length scale, in km
        tstar (np.array): characteristic speed scale, in km/sec

    Returns:
        (np.array): length 6 1D array of Cartesian state, non-dimensional
    """

    out = np.zeros((6,))
    out[0] = state[0] / lstar
    out[1] = state[1] / lstar
    out[2] = state[2] / lstar
    out[3] = state[3] / (lstar/tstar)
    out[4] = state[4] / (lstar/tstar)
    out[5] = state[5] / (lstar/tstar)

    return out


# convert state from canonical to dimensional
@jit(nopython=True)
def nondim2dim(state, lstar, tstar):
    """Convert from non-dimensional to dimensional state
    
    Args:
        state (np.array): length 6 1D array of Cartesian state, non-dimensional
        lstar (np.array): characteristic length scale, in km
        tstar (np.array): characteristic speed scale, in km/sec

    Returns:
        (np.array): length 6 1D array of Cartesian state, in units km, km/sec
    """

    out = np.zeros((6,))
    out[0] = state[0] * lstar
    out[1] = state[1] * lstar
    out[2] = state[2] * lstar
    out[3] = state[3] * (lstar/tstar)
    out[4] = state[4] * (lstar/tstar)
    out[5] = state[5] * (lstar/tstar)

    return out


# shifting state linearly along an axis
@jit(nopython=True)
def shift_state(state, shift, axis="x"):
    """State is shifted along the x-axis by input amount

    Args:
        state (np.array): 1D array of state
        shift (float): value to shift along user-defined axis
        axis (str): axis, i.e. "x", "y", "z"

    Returns:
        (np.array): shifted 1D array of state
    """
    
    out = np.zeros((len(state),))

    for j in range(len(state)):
        out[j] = state[j]

    if axis == "x":
        out[0] = out[0] + shift
    elif axis == "y":
        out[1] = out[1] + shift
    elif axis == "z":
        out[2] = out[2] + shift
            
    return out


# converting rotating state to inertial state
@jit(nopython=True)
def rotating2inertial(state_r, theta):
    """Converts state in rotating frame state to intrertial frame (assumes rotating frame rotates anti-clockwise)
    
    Args:
        state_r (np.array): 1D array of state in rotating frame
        theta (float): rotation angle, in radians


    Returns:
        (np.array): 1D arrat of state in inertial frame
    """

    # construct transformation matrix
    rotmat = np.array([ [np.cos(theta), -np.sin(theta), 0.0, 0.0, 0.0, 0.0],
                        [np.sin(theta),  np.cos(theta), 0.0, 0.0, 0.0, 0.0],
                        [0.0,    0.0,    1.0,                0.0, 0.0, 0.0],
                        [-np.sin(theta), -np.cos(theta), 0.0, np.cos(theta), -np.sin(theta), 0.0],
                        [np.cos(theta),  -np.sin(theta), 0.0, np.sin(theta),  np.cos(theta), 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] ])

    # transform
    state_i = np.dot(rotmat, state_r)

    return state_i


# converting inertial state to rotating state
@jit(nopython=True)
def inertial2rotating(state_i, theta):
    """Converts state in inertial frame to rotating frame
    
    Args:
        state_i (np.array): 1D array of state in inertial frame
        theta (float): rotation angle

    Returns:
        (np.array): 1D arrat of state in rotational frame
    """

    # construct transformation matrix
    rotmat = la.inv( np.array([ [np.cos(theta), -np.sin(theta), 0.0, 0.0, 0.0, 0.0],
                                [np.sin(theta), np.cos(theta), 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0],
                                [-np.sin(theta), -np.cos(theta), 0.0, np.cos(theta), -np.sin(theta), 0.0],
                                [np.cos(theta), -np.sin(theta), 0.0, np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0] ]) )

    # transform
    state_r = np.dot(rotmat, state_i)

    return state_r


# transform from EM rotating barycenter frame to SE rotating barycenter frame
@jit(nopython=True)
def transform_EMcr3bp_to_SEcr3bp(beta, t_EM, state_EM, mu_EM, mu_SE, Lstar_EM, Tstar_EM, Lstar_SE, Tstar_SE):
    """Function to transform state vector from Earth-Moon CR3BP rotating frame to Sun-Earth CR3BP rotating frame
    
    Args:
        beta (float): angle to rotate, anti-clockwise in Earth-Moon frame, in radians
        t_EM (float): time, in Earth-Moon frame
        state_EM (np.array): state in Earth-Moon CR3BP rotating frame
        mu_EM (float): Earth-Moon mass parameter
        mu_SE (float): Sun-Earth mass parameter
        Lstar_EM (float): Earth-Moon characteristic length scale
        Tstar_EM (float): Earth-Moon characteristic time scale
        Lstar_SE (float): Sun-Earth characteristic length scale
        Tstar_SE (float): Sun-Earth characteristic time scale
    Returns:
        (np.array): state in Sun-Earth CR3BP rotating frame
    """
    # Earth-centered, rotating frame
    state_EM_Ecentered = state_EM + np.array([ mu_EM , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ])
    # Earth-centered, Earth-Moon inertial frame
    theta_EM = t_EM + beta
    state_EM_inertial = rotating2inertial(state_EM_Ecentered, theta_EM)
    
    # re-scaling factors
    lambda_L = Lstar_EM / Lstar_SE
    lambda_T = Tstar_EM / Tstar_SE
    lambda_V = lambda_L / lambda_T
    
    # re-scale position and velocity
    r_SE = lambda_L * state_EM_inertial[0:3]
    v_SE = lambda_V * state_EM_inertial[3:6]
    state_SE_inertial = np.concatenate((r_SE, v_SE))
    
    # Earth-centered, Sun-Earth rotating frame
    theta_SE = lambda_T * t_EM   # - beta
    
    state_SE = inertial2rotating( state_SE_inertial, theta_SE ) + np.array([ 1-mu_SE , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ])
    
    return state_SE


# transform from SE rotating barycenter frame to EM rotating barycenter frame
@jit(nopython=True)
def transform_SEcr3bp_to_EMcr3bp(beta, t_SE, state_SE, mu_EM, mu_SE, Lstar_EM, Tstar_EM, Lstar_SE, Tstar_SE):
    """Function to transform state vector from Sun-Earth CR3BP rotating frame to Earht-Moon CR3BP rotating frame
    
    Args:
        beta (float): angle to rotate, anti-clockwise in Earth-Moon frame, in radians
        t_SE (float): time, in Sun-Earth frame
        state_EM (np.array): state in Earth-Moon CR3BP rotating frame
        mu_EM (float): Earth-Moon mass parameter
        mu_SE (float): Sun-Earth mass parameter
        Lstar_EM (float): Earth-Moon characteristic length scale
        Tstar_EM (float): Earth-Moon characteristic time scale
        Lstar_SE (float): Sun-Earth characteristic length scale
        Tstar_SE (float): Sun-Earth characteristic time scale
    Returns:
        (np.array): state in Sun-Earth CR3BP rotating frame
    """
    
    # Earth-centered, Sun-Earth, rotating frame
    state_SE_Ecentered = state_SE - np.array([ 1-mu_SE , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ])
    # Earth-centered, Sun-Earth inertial frame
    beta_SE = t_SE
    state_SE_inertial = rotating2inertial(state_SE_Ecentered, beta_SE)
    
    #re-scaling factors
    lambda_L = Lstar_SE / Lstar_EM
    lambda_T = Tstar_SE / Tstar_EM
    lambda_V = lambda_L / lambda_T
    
    # re-scale position and velocity
    r_EM = lambda_L * state_SE_inertial[0:3]
    v_EM = lambda_V * state_SE_inertial[3:6]
    state_EM_inertial = np.concatenate((r_EM, v_EM))
    
    # Earth-centered, Earth-Moon rotating frame -> Barycenter, Earth-Moon rotating frame
    theta_EM = lambda_T * t_SE + beta
    state_EM = inertial2rotating( state_EM_inertial, theta_EM ) + np.array([ -mu_EM , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ])
    
    return state_EM


# ------------------------------------------------------------------------------- #
# function to convert propagation output from rotating to inertial
def convert_propout_rotating2inertial(propout, theta0, mu, mass_center=1):
    """Converts propagation output from rotating to inertial, centered around mass 1 or mass 2
    
    Args:
        propout (dict): propagation output from prop.propagate_xx functions
        theta0 (float): initial phase angle, added to the time array entries, in radians
        mu (float): CR3BP mass parameter
        mass_center (int): location of center, either mass 1 or mass 2 (1 or 2)

    Returns:
        (dict): dictionary of transformed propagation output
    """
    # call function to convert
    x_tmp, y_tmp, z_tmp, vx_tmp, vy_tmp, vz_tmp = _convertsays_rotating2inertial(propout["xs"], propout["ys"], propout["zs"], 
                                propout["vxs"], propout["vys"], propout["vzs"], propout["times"], mu, theta0, mass_center)
    # initial state
    statef0  = np.array([ x_tmp[0] , y_tmp[0] , z_tmp[0] , 
                           vx_tmp[0] , vy_tmp[0] , vz_tmp[0] ])
    # final state
    statef_i = np.array([ x_tmp[-1] , y_tmp[-1] , z_tmp[-1] , 
                           vx_tmp[-1] , vy_tmp[-1] , vz_tmp[-1] ])
    # construct output dictionary 
    propout_i = {"xs": x_tmp, "ys": y_tmp, "zs": z_tmp, 
                  "vxs": vx_tmp, "vys": vy_tmp, "vzs": vz_tmp,
                  "times": propout["times"], 'state0': state0, 'statef': statef_i}

    return propout_i


@jit(nopython=True) 
def _convertsays_rotating2inertial(xs, ys, zs, vxs, vys, vzs, times, mu, theta0, mass_center):
    """Conversion from rotating to inertial frame wrapped with jit decorator"""
    # initialize array
    x_tmp  = np.zeros((len(times),))
    y_tmp  = np.zeros((len(times),))
    z_tmp  = np.zeros((len(times),))
    vx_tmp = np.zeros((len(times),))
    vy_tmp = np.zeros((len(times),))
    vz_tmp = np.zeros((len(times),))
    # iterate over each element
    for idx in range(len(times)):
        # state in rotating frame, centered at the body
        if mass_center==1:
            state_rot = np.array([ xs[idx] + mu, ys[idx], 
                                   zs[idx],      vxs[idx], 
                                   vys[idx],     vzs[idx] ])
        else:
            state_rot = np.array([ xs[idx] - (1 - mu), ys[idx],     # FIXME --- should be (1-mu)??? changed from (1+mu) on 2021/01/05 
                                   zs[idx],            vxs[idx], 
                                   vys[idx],           vzs[idx] ])
        theta_rot = times[idx] + theta0
        # convert state
        state_i = rotating2inertial(state_rot, theta_rot)
        # store position and velocity
        x_tmp[idx]  = state_i[0]
        y_tmp[idx]  = state_i[1]
        z_tmp[idx]  = state_i[2]
        vx_tmp[idx] = state_i[3]
        vy_tmp[idx] = state_i[4]
        vz_tmp[idx] = state_i[5]
    return x_tmp, y_tmp, z_tmp, vx_tmp, vy_tmp, vz_tmp