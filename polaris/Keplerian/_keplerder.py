"""
Kepler's equation with Der's formulation for computing state-transition matrix, from G. Der (1996). An elegant state transition matrix

Usage:
	statef, stmf = keplerder()
"""

import numpy as np
from numba import jit
import numpy.linalg as la

try:
    from ._orbitalElements import get_eccentricity, get_semiMajorAxis
except:
    from _orbitalElements import get_eccentricity, get_semiMajorAxis


# ------------------------------------------------------------------------------------- #
# hyperbolic trig functions
@jit(nopython=True)
def hypertrig_s(z):
    if z > 0.0:
        s = (np.sqrt(z)-np.sin(np.sqrt(z))) / (np.sqrt(z))**3
    elif z < 0.0:
        s = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z))**3
    else:
        s = 1.0/6.0
    return s

@jit(nopython=True)
def hypertrig_c(z):
    if z > 0.0:
        c = (1.0 - np.cos(np.sqrt(z)))/z
    elif z < 0.0:
        c = (np.cosh(-z) - 1)/(-z)
    else:
        c = 0.5
    return c


# ------------------------------------------------------------------------------------- #
# Lagrange parameter functions
@jit(nopython=True)
def lagrange_us(x, alpha):
    """Function computes U0 ~ U3 from G. Der 1996"""
    # evaluate hypertrig function
    S = hypertrig_s(alpha*x**2)
    C = hypertrig_c(alpha*x**2)
    # parameters
    u0 = 1 - alpha * x**2 * C
    u1 = x*(1 - alpha * x**2 * S)
    u2 = x**2 * C
    u3 = x**3 * S
    return u0, u1, u2, u3


@jit(nopython=True)
def kepler_der(x, alpha, t, t0, mu, r0, sigma0):
    """Function computes Kepler's time equation and its derivatives in G. Der 1996 form"""
    u0, u1, u2, u3 = lagrange_us(x, alpha)
    Fun = r0*u1 + sigma0*u2 + u3 - np.sqrt(mu) * (t - t0)
    dF  = r0*u0 + sigma0*u1 + u2
    d2F = sigma0*u0 + (1 - alpha*r0)*u1
    return Fun, dF, d2F


@jit(nopython=True)
def lagrange_coefficients_der(mu, x, alpha, r0, v0, sigma0, u0, u1, u2, u3, r, sigma):
    """Function computes Lagrange coefficients (as defined in G. Der 1996)"""
    # evaluate hypertrig function
    S = hypertrig_s(alpha*x**2)
    C = hypertrig_c(alpha*x**2)
    # scalar function
    f = 1.0 - u2/r0
    g = r0*u1 / np.sqrt(mu) + sigma0*u2 / np.sqrt(mu)
    fdot = -np.sqrt(mu) / (r*r0) * u1
    gdot = 1.0 - u2/r
    return f, g, fdot, gdot


# ------------------------------------------------------------------------------------- #
# STM coefficients (eqn. 18)
@jit(nopython=True)
def der_stm_coefs(mu, alpha, r0, v0, r, u0, u1, u2, u3, sigma0, sigma):
    """Scalar coefficients of STM dyads"""
    # first row coefficients
    c11 = 1/(alpha*r*r0**2) * (3*u1*u3 + (alpha*r0 - 2)*u2**2) + u1**2/r + u2/r0
    c12 = v0*u1*u2 / (r*np.sqrt(mu))
    c13 = v0/(alpha*r*r0**2*np.sqrt(mu))*(r0*u1*u2 + 2*sigma0*u2**2 + 3*u2*u3 - 3*r*u3 + alpha*r0**2*u1*u2)
    c14 = v0**2 * u2**2 / (r*mu)
    # second row
    c21 = r0*u1*u2 / (r*np.sqrt(mu))
    c22 = v0/(alpha*r*mu) * (3*u1*u3 + (alpha*r0 - 2)*u2**2)
    c23 = r0*v0*u2**2 / (r*mu)
    c24 = v0**2 / (alpha*r*mu*np.sqrt(mu)) * (r0*u1*u2 + 2*sigma0*u2**2 + 3*u2*u3 - 3*r*u3)
    # third row
    c31 = np.sqrt(mu)/(alpha*r**3*r0**2) * (r*(3*u0*u3 - u1*u2 + 2*alpha*r0*u1*u2) - sigma*(3*u1*u3 - 2*u2**2 + alpha*r0*u2**2)) + np.sqrt(mu)/(r**3*r0)*(2*r*r0*u0*u1 + r**2*u1 - sigma*r0*u1**2)
    c32 = v0/r**3 * (r*(u0*u2 + u1**2) - sigma*u1*u2)
    c33 = -3*v0*u2/(alpha*r*r0**2) + v0/(alpha*r**2*r0**2) * (4*sigma0*u1*u2 + r0*(u0*u2 + u1**2) - 3*(u1*u3 + u2**2)) - sigma*v0*u2 / (alpha*r**3*r0**2)*(r0*u1 + 2*sigma0*u2 + 3*u3) + v0/r**3*(r*(u0*u2 + u1**2) - sigma*u1*u2)
    c34 = v0**2 / (r**3*np.sqrt(mu)) * (2*r*u1*u2 - sigma*u2**2)
    # fourth row
    c41 = r0/r**3 * (r*(u0*u2 + u1**2) - sigma*u1*u2)
    c42 = v0/(alpha*r**3*np.sqrt(mu)) * (r*(3*u0*u3 - u1*u2 + 2*alpha*r0*u1*u2) - sigma*(3*u1*u3 - 2*u2**2 + alpha*r0*u2**2))
    c43 = r0*v0/(r**3*np.sqrt(mu)) * (2*r*u1*u2 - sigma*u2**2)
    c44 = -3*v0**2*u2 / (alpha*r*mu) + v0**2 / (alpha*r**2*mu) * (4*sigma0*u1*u2 + r0*(u0*u2 + u1**2) + 3*(u1*u3 + u2**2)) - sigma*v0**2*u2/(alpha*r**3*mu) * (r0*u1 + 2*sigma0*u2 + 3*u3)
    return c11,c12,c13,c14,c21,c22,c23,c24,c31,c32,c33,c34,c41,c42,c43,c44


# ------------------------------------------------------------------------------------- #
# Keplerian time-propagation with G. Der formulation
@jit(nopython=True)
def keplerder(mu, state0, t0, t, tol=1e-12, maxiter=10, get_stm=False):
    """Function computes position at some future time t

    Args:
		mu (float): gravitational parameter
		state0 (np.array): initial state [x,y,z,vx,vy,vz]
		t0 (float): initial time
		t (float): final time
		tol (float): tolerance on Laguerre-correction
		maxiter (int): max allowed iteration allowed for Laguerre-correction
        get_stm (bool): whether to get STM; if False, returns (6,6) zeros matrix

    Returns:
		(tuple): final state, final STM
    """
    # ------------------------------------------ #
    # SET-UP PROBLEM
    r0, v0 = state0[0:3], state0[3:6]
    sma = get_semiMajorAxis(state0, mu)
    alpha = 1.0/sma
    sigma0 =  np.dot(r0, v0) / np.sqrt(mu)
    
    # ------------------------------------------ #
    # ITERATION WITH LAGUERRE-CONWAY METHOD
    # initial guess
    ecc = la.norm(get_eccentricity(state0, mu))
    if ecc < 1:
        x0 = alpha*np.sqrt(mu)*(t-t0)
    else:
        x0 = np.sqrt(mu)*(t-t0)/(10*la.norm(r0))
    # initialize iteratation
    count = 0
    while count < maxiter:
        # evaluate function
        Fun, dF, d2F =  kepler_der(x0, alpha, t, t0, mu, la.norm(r0), sigma0)
        if np.abs(Fun) < tol:
            break
        # Laguerre-Conway iteration
        x1 = x0 - 5*Fun / (dF + dF/np.abs(dF) * np.sqrt(abs(16*dF**2 - 20*Fun*d2F)) )   
        count += 1
        x0 = x1
    if np.abs(Fun) > tol:
        print("Laguerre-Conway failed! error: ")
    
    # ------------------------------------------ #
    # COMPUTE FINAL POSITION
    # convert back to position
    u0, u1, u2, u3 = lagrange_us(x1, alpha)
    r_scal = la.norm(r0)*u0 + sigma0*u1 + u2
    sigma = sigma0*u0 + (1-alpha*la.norm(r0))*u1
    
    # get lagrange coefficients
    f, g, fdot, gdot = lagrange_coefficients_der(mu, x1, alpha, la.norm(r0), la.norm(v0), sigma0, u0, u1, u2, u3, r_scal, sigma)
    # create map for state
    fullmap = np.array([[f, 0., 0., g, 0., 0.], [0., f, 0., 0., g, 0.], [0., 0., f, 0., 0., g],
          [fdot, 0., 0., gdot, 0., 0.], [0., fdot, 0., 0., gdot, 0.], [0., 0., fdot, 0., 0., gdot]])
    state1 = np.dot(fullmap, state0)
    
    # ------------------------------------------ #
    # COMPUTE FINAL STM
    if get_stm==True:
        # submatrices M's
        M1 = r0.reshape(3,1) * r0.reshape(1,3) / la.norm(r0)**2
        M2 = r0.reshape(3,1) * v0.reshape(1,3) / (la.norm(r0)*la.norm(v0))
        M3 = v0.reshape(3,1) * r0.reshape(1,3) / (la.norm(r0)*la.norm(v0))
        M4 = v0.reshape(3,1) * v0.reshape(1,3) / la.norm(v0)**2
        # coefficients
        c11,c12,c13,c14,c21,c22,c23,c24,c31,c32,c33,c34,c41,c42,c43,c44 = der_stm_coefs(mu, alpha, la.norm(r0), 
                                                        la.norm(v0), r_scal, u0, u1, u2, u3, sigma0, sigma)
        # construct STM
        Rtilda = f*np.identity(3)    + c11*M1 + c12*M2 + c13*M3 + c14*M4
        R      = g*np.identity(3)    + c21*M1 + c22*M2 + c23*M3 + c24*M4
        Vtilda = fdot*np.identity(3) + c31*M1 + c32*M2 + c33*M3 + c34*M4
        V      = gdot*np.identity(3) + c41*M1 + c42*M2 + c43*M3 + c44*M4
        stm = np.concatenate(( np.concatenate((Rtilda,R), axis=1), np.concatenate((Vtilda,V), axis=1) ), axis=0 )
    else:
        stm = np.zeros((6,6))
    return state1, stm


# Keplerian time-propagation with G. Der formulation, with no STM
@jit(nopython=True)
def keplerder_nostm(mu, state0, t0, t, tol=1e-12, maxiter=10):
    """Function computes position at some future time t

    Args:
        mu (float): gravitational parameter
        state0 (np.array): initial state [x,y,z,vx,vy,vz]
        t0 (float): initial time
        t (float): final time
        tol (float): tolerance on Laguerre-correction
        maxiter (int): max allowed iteration allowed for Laguerre-correction

    Returns:
        (tuple): final state, final STM
    """
    # ------------------------------------------ #
    # SET-UP PROBLEM
    r0, v0 = state0[0:3], state0[3:6]
    sma = get_semiMajorAxis(state0, mu)
    alpha = 1.0/sma
    sigma0 =  np.dot(r0, v0) / np.sqrt(mu)
    
    # ------------------------------------------ #
    # ITERATION WITH LAGUERRE-CONWAY METHOD
    # initial guess
    ecc = la.norm(get_eccentricity(state0, mu))
    if ecc < 1:
        x0 = alpha*np.sqrt(mu)*(t-t0)
    else:
        x0 = np.sqrt(mu)*(t-t0)/(10*la.norm(r0))
    # initialize iteratation
    count = 0
    while count < maxiter:
        # evaluate function
        Fun, dF, d2F =  kepler_der(x0, alpha, t, t0, mu, la.norm(r0), sigma0)
        if np.abs(Fun) < tol:
            break
        # Laguerre-Conway iteration
        x1 = x0 - 5*Fun / (dF + dF/np.abs(dF) * np.sqrt(abs(16*dF**2 - 20*Fun*d2F)) )   
        count += 1
        x0 = x1
    if np.abs(Fun) > tol:
        print("Laguerre-Conway failed! error: ")
    
    # ------------------------------------------ #
    # COMPUTE FINAL POSITION
    # convert back to position
    u0, u1, u2, u3 = lagrange_us(x1, alpha)
    r_scal = la.norm(r0)*u0 + sigma0*u1 + u2
    sigma = sigma0*u0 + (1-alpha*la.norm(r0))*u1
    
    # get lagrange coefficients
    f, g, fdot, gdot = lagrange_coefficients_der(mu, x1, alpha, la.norm(r0), la.norm(v0), sigma0, u0, u1, u2, u3, r_scal, sigma)
    # create map for state
    fullmap = np.array([[f, 0., 0., g, 0., 0.], [0., f, 0., 0., g, 0.], [0., 0., f, 0., 0., g],
          [fdot, 0., 0., gdot, 0., 0.], [0., fdot, 0., 0., gdot, 0.], [0., 0., fdot, 0., 0., gdot]])
    state1 = np.dot(fullmap, state0)
    return state1



# # ------------------------------------------------------------------------------------- #
# # Keplerian time-propagation with G. Der formulation
# @jit(nopython=True)
# def keplerder_old(mu, state0, t0, t, tol=1e-12, maxiter=10):
#     """Function computes position at some future time t

#     Args:
#         mu (float): gravitational parameter
#         state0 (np.array): initial state [x,y,z,vx,vy,vz]
#         t0 (float): initial time
#         t (float): final time
#         tol (float): tolerance on Laguerre-correction
#         maxiter (int): max allowed iteration allowed for Laguerre-correction

#     Returns:
#         (tuple): final state, final STM
#     """
#     # ------------------------------------------ #
#     # SET-UP PROBLEM
#     r0, v0 = state0[0:3], state0[3:6]
#     sma = get_semiMajorAxis(state0, mu)
#     alpha = 1.0/sma
#     sigma0 =  np.dot(r0, v0) / np.sqrt(mu)
    
#     # ------------------------------------------ #
#     # ITERATION WITH LAGUERRE-CONWAY METHOD
#     # initial guess
#     x0 = alpha*np.sqrt(mu)*(t-t0)
#     # initialize iteratation
#     count = 0
#     while count < maxiter:
#         # evaluate function
#         Fun, dF, d2F =  kepler_der(x0, alpha, t, t0, mu, la.norm(r0), sigma0)
#         if np.abs(Fun) < tol:
#             break
#         # Laguerre-Conway iteration
#         x1 = x0 - 5*Fun / (dF + dF/np.abs(dF) * np.sqrt(16*dF**2 - 20*Fun*d2F) )   
#         count += 1
#         x0 = x1
    
#     # ------------------------------------------ #
#     # COMPUTE FINAL POSITION
#     # convert back to position
#     u0, u1, u2, u3 = lagrange_us(x1, alpha)
#     r_scal = la.norm(r0)*u0 + sigma0*u1 + u2
#     sigma = sigma0*u0 + (1-alpha*la.norm(r0))*u1
    
#     # get lagrange coefficients
#     f, g, fdot, gdot = lagrange_coefficients_der(mu, x1, alpha, la.norm(r0), la.norm(v0), sigma0, u0, u1, u2, u3, r_scal, sigma)
#     # create map for state
#     rmap = np.concatenate((f*np.identity(3) , g*np.identity(3)), axis=1)
#     vmap = np.concatenate((fdot*np.identity(3) , gdot*np.identity(3)), axis=1)
#     fullmap = np.concatenate((rmap, vmap), axis=0)
#     state1 = np.dot(fullmap, state0)
    
#     # ------------------------------------------ #
#     # COMPUTE FINAL STM
#     # submatrices M's
#     M1 = r0.reshape(3,1) * r0.reshape(1,3) / la.norm(r0)**2
#     M2 = r0.reshape(3,1) * v0.reshape(1,3) / (la.norm(r0)*la.norm(v0))
#     M3 = v0.reshape(3,1) * r0.reshape(1,3) / (la.norm(r0)*la.norm(v0))
#     M4 = v0.reshape(3,1) * v0.reshape(1,3) / la.norm(v0)**2
#     # coefficients
#     c11,c12,c13,c14,c21,c22,c23,c24,c31,c32,c33,c34,c41,c42,c43,c44 = der_stm_coefs(mu, alpha, la.norm(r0), 
#                                                     la.norm(v0), r_scal, u0, u1, u2, u3, sigma0, sigma)
#     # construct STM
#     Rtilda = f*np.identity(3)    + c11*M1 + c12*M2 + c13*M3 + c14*M4
#     R      = g*np.identity(3)    + c21*M1 + c22*M2 + c23*M3 + c24*M4
#     Vtilda = fdot*np.identity(3) + c31*M1 + c32*M2 + c33*M3 + c34*M4
#     V      = gdot*np.identity(3) + c41*M1 + c42*M2 + c43*M3 + c44*M4
#     stm = np.concatenate(( np.concatenate((Rtilda,R), axis=1), np.concatenate((Vtilda,V), axis=1) ), axis=0 )
#     return state1, stm
