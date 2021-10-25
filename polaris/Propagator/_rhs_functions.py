"""
Right-hand side functions of equations of motions
"""

import numpy as np
import numpy.linalg as la
from numba import jit


# ------------------------------------------------------------------------------------------------ #
# define RHS function in two body problem
@jit(nopython=True)
def rhs_twobody(t, state, gm):
    """Equation of motion in two body problem, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        gm (float): two body mass parameter
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    dstate = np.zeros(
        6,
    )
    # position
    dstate[0] = state[3]
    dstate[1] = state[4]
    dstate[2] = state[5]
    # velocities
    dstate[3] = -(gm / r ** 3) * state[0]
    dstate[4] = -(gm / r ** 3) * state[1]
    dstate[5] = -(gm / r ** 3) * state[2]
    return dstate


# ------------------------------------------------------------------------------------------------ #
# define RHS function with STMin two body problem
@jit(nopython=True)
def rhs_twobody_with_STM(t, state, mu):
    # radius
    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    dstate = np.zeros(
        len(state),
    )
    # positions
    dstate[0] = state[3]
    dstate[1] = state[4]
    dstate[2] = state[5]
    # velocities
    dstate[3] = -(mu / r ** 3) * state[0]
    dstate[4] = -(mu / r ** 3) * state[1]
    dstate[5] = -(mu / r ** 3) * state[2]
    # gravity gradient entries
    g00 = (mu / r ** 3) * (-1 + 3 * state[0] ** 2 / r ** 2)
    g01 = (mu / r ** 3) * (3 * state[0] * state[1] / r ** 2)
    g02 = (mu / r ** 3) * (3 * state[0] * state[2] / r ** 2)
    g11 = (mu / r ** 3) * (-1 + 3 * state[1] ** 2 / r ** 2)
    g12 = (mu / r ** 3) * (3 * state[1] * state[2] / r ** 2)
    g22 = (mu / r ** 3) * (-1 + 3 * state[2] ** 2 / r ** 2)

    # a-matrix entries
    a00, a01, a02, a03, a04, a05 = 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    a10, a11, a12, a13, a14, a15 = 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    a20, a21, a22, a23, a24, a25 = 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    a30, a31, a32, a33, a34, a35 = g00, g01, g02, 0.0, 0.0, 0.0
    a40, a41, a42, a43, a44, a45 = g01, g11, g12, 0.0, 0.0, 0.0
    a50, a51, a52, a53, a54, a55 = g02, g12, g22, 0.0, 0.0, 0.0

    # derivative of STM
    # 1st row ...
    dstate[6] = (
        a00 * state[6]
        + a01 * state[12]
        + a02 * state[18]
        + a03 * state[24]
        + a04 * state[30]
        + a05 * state[36]
    )
    dstate[7] = (
        a00 * state[7]
        + a01 * state[13]
        + a02 * state[19]
        + a03 * state[25]
        + a04 * state[31]
        + a05 * state[37]
    )
    dstate[8] = (
        a00 * state[8]
        + a01 * state[14]
        + a02 * state[20]
        + a03 * state[26]
        + a04 * state[32]
        + a05 * state[38]
    )
    dstate[9] = (
        a00 * state[9]
        + a01 * state[15]
        + a02 * state[21]
        + a03 * state[27]
        + a04 * state[33]
        + a05 * state[39]
    )
    dstate[10] = (
        a00 * state[10]
        + a01 * state[16]
        + a02 * state[22]
        + a03 * state[28]
        + a04 * state[34]
        + a05 * state[40]
    )
    dstate[11] = (
        a00 * state[11]
        + a01 * state[17]
        + a02 * state[23]
        + a03 * state[29]
        + a04 * state[35]
        + a05 * state[41]
    )

    # 2nd row ...
    dstate[12] = (
        a10 * state[6]
        + a11 * state[12]
        + a12 * state[18]
        + a13 * state[24]
        + a14 * state[30]
        + a15 * state[36]
    )
    dstate[13] = (
        a10 * state[7]
        + a11 * state[13]
        + a12 * state[19]
        + a13 * state[25]
        + a14 * state[31]
        + a15 * state[37]
    )
    dstate[14] = (
        a10 * state[8]
        + a11 * state[14]
        + a12 * state[20]
        + a13 * state[26]
        + a14 * state[32]
        + a15 * state[38]
    )
    dstate[15] = (
        a10 * state[9]
        + a11 * state[15]
        + a12 * state[21]
        + a13 * state[27]
        + a14 * state[33]
        + a15 * state[39]
    )
    dstate[16] = (
        a10 * state[10]
        + a11 * state[16]
        + a12 * state[22]
        + a13 * state[28]
        + a14 * state[34]
        + a15 * state[40]
    )
    dstate[17] = (
        a10 * state[11]
        + a11 * state[17]
        + a12 * state[23]
        + a13 * state[29]
        + a14 * state[35]
        + a15 * state[41]
    )

    # 3rd row ...
    dstate[18] = (
        a20 * state[6]
        + a21 * state[12]
        + a22 * state[18]
        + a23 * state[24]
        + a24 * state[30]
        + a25 * state[36]
    )
    dstate[19] = (
        a20 * state[7]
        + a21 * state[13]
        + a22 * state[19]
        + a23 * state[25]
        + a24 * state[31]
        + a25 * state[37]
    )
    dstate[20] = (
        a20 * state[8]
        + a21 * state[14]
        + a22 * state[20]
        + a23 * state[26]
        + a24 * state[32]
        + a25 * state[38]
    )
    dstate[21] = (
        a20 * state[9]
        + a21 * state[15]
        + a22 * state[21]
        + a23 * state[27]
        + a24 * state[33]
        + a25 * state[39]
    )
    dstate[22] = (
        a20 * state[10]
        + a21 * state[16]
        + a22 * state[22]
        + a23 * state[28]
        + a24 * state[34]
        + a25 * state[40]
    )
    dstate[23] = (
        a20 * state[11]
        + a21 * state[17]
        + a22 * state[23]
        + a23 * state[29]
        + a24 * state[35]
        + a25 * state[41]
    )

    # 4th row ...
    dstate[24] = (
        a30 * state[6]
        + a31 * state[12]
        + a32 * state[18]
        + a33 * state[24]
        + a34 * state[30]
        + a35 * state[36]
    )
    dstate[25] = (
        a30 * state[7]
        + a31 * state[13]
        + a32 * state[19]
        + a33 * state[25]
        + a34 * state[31]
        + a35 * state[37]
    )
    dstate[26] = (
        a30 * state[8]
        + a31 * state[14]
        + a32 * state[20]
        + a33 * state[26]
        + a34 * state[32]
        + a35 * state[38]
    )
    dstate[27] = (
        a30 * state[9]
        + a31 * state[15]
        + a32 * state[21]
        + a33 * state[27]
        + a34 * state[33]
        + a35 * state[39]
    )
    dstate[28] = (
        a30 * state[10]
        + a31 * state[16]
        + a32 * state[22]
        + a33 * state[28]
        + a34 * state[34]
        + a35 * state[40]
    )
    dstate[29] = (
        a30 * state[11]
        + a31 * state[17]
        + a32 * state[23]
        + a33 * state[29]
        + a34 * state[35]
        + a35 * state[41]
    )

    # 5th row ...
    dstate[30] = (
        a40 * state[6]
        + a41 * state[12]
        + a42 * state[18]
        + a43 * state[24]
        + a44 * state[30]
        + a45 * state[36]
    )
    dstate[31] = (
        a40 * state[7]
        + a41 * state[13]
        + a42 * state[19]
        + a43 * state[25]
        + a44 * state[31]
        + a45 * state[37]
    )
    dstate[32] = (
        a40 * state[8]
        + a41 * state[14]
        + a42 * state[20]
        + a43 * state[26]
        + a44 * state[32]
        + a45 * state[38]
    )
    dstate[33] = (
        a40 * state[9]
        + a41 * state[15]
        + a42 * state[21]
        + a43 * state[27]
        + a44 * state[33]
        + a45 * state[39]
    )
    dstate[34] = (
        a40 * state[10]
        + a41 * state[16]
        + a42 * state[22]
        + a43 * state[28]
        + a44 * state[34]
        + a45 * state[40]
    )
    dstate[35] = (
        a40 * state[11]
        + a41 * state[17]
        + a42 * state[23]
        + a43 * state[29]
        + a44 * state[35]
        + a45 * state[41]
    )

    # 6th row ...
    dstate[36] = (
        a50 * state[6]
        + a51 * state[12]
        + a52 * state[18]
        + a53 * state[24]
        + a54 * state[30]
        + a55 * state[36]
    )
    dstate[37] = (
        a50 * state[7]
        + a51 * state[13]
        + a52 * state[19]
        + a53 * state[25]
        + a54 * state[31]
        + a55 * state[37]
    )
    dstate[38] = (
        a50 * state[8]
        + a51 * state[14]
        + a52 * state[20]
        + a53 * state[26]
        + a54 * state[32]
        + a55 * state[38]
    )
    dstate[39] = (
        a50 * state[9]
        + a51 * state[15]
        + a52 * state[21]
        + a53 * state[27]
        + a54 * state[33]
        + a55 * state[39]
    )
    dstate[40] = (
        a50 * state[10]
        + a51 * state[16]
        + a52 * state[22]
        + a53 * state[28]
        + a54 * state[34]
        + a55 * state[40]
    )
    dstate[41] = (
        a50 * state[11]
        + a51 * state[17]
        + a52 * state[23]
        + a53 * state[29]
        + a54 * state[35]
        + a55 * state[41]
    )

    return dstate


# ------------------------------------------------------------------------------------------------ #
# define RHS function in two body problem with J2
@jit(nopython=True)
def rhs_twobody_j2(t, state, gm, j2, re):
    """Equation of motion in two body problem, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        gm (float): two body mass parameter
        j2 (float): J2 coefficient
        re (float): radius of reference body

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    dstate = np.zeros(
        6,
    )

    # common multiplier of J2 term acceleration
    J2mlt = (3 / 2) * j2 * (gm / r ** 2) * (re / r) ** 2  # scalar
    # perturbing accelerations
    p = [
        J2mlt * (5 * state[2] ** 2 / r ** 2 - 1) * state[0] / r,
        J2mlt * (5 * state[2] ** 2 / r ** 2 - 1) * state[1] / r,
        J2mlt * (5 * state[2] ** 2 / r ** 2 - 3) * state[2] / r,
    ]

    # position
    dstate[0] = state[3]
    dstate[1] = state[4]
    dstate[2] = state[5]
    # velocities
    dstate[3] = -(gm / r ** 3) * state[0] + p[0]
    dstate[4] = -(gm / r ** 3) * state[1] + p[1]
    dstate[5] = -(gm / r ** 3) * state[2] + p[2]
    return dstate


# ------------------------------------------------------------------------------------------------ #
# define RHS function in CR3BP
@jit(nopython=True)
def rhs_cr3bp(t, state, mu):
    """Equation of motion in CR3BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """

    # unpack positions
    x = state[0]
    y = state[1]
    z = state[2]
    # unpack velocities
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((6,))
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] = (
        2 * vy + x - ((1 - mu) / r1 ** 3) * (mu + x) + (mu / r2 ** 3) * (1 - mu - x)
    )
    deriv[4] = -2 * vx + y - ((1 - mu) / r1 ** 3) * y - (mu / r2 ** 3) * y
    deriv[5] = -((1 - mu) / r1 ** 3) * z - (mu / r2 ** 3) * z

    return deriv


# ------------------------------------------------------------------------------------------------ #
# define RHS function in CR3BP, in 2D
@jit(nopython=True)
def rhs_pcr3bp(t, state, mu):
    """Equation of motion in planar CR3BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 4
        mu (float): CR3BP parameter
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """

    # unpack positions
    x = state[0]
    y = state[1]
    # unpack velocities
    vx = state[2]
    vy = state[3]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((4,))
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    # velocity derivatives
    deriv[2] = (
        2 * vy + x - ((1 - mu) / r1 ** 3) * (mu + x) + (mu / r2 ** 3) * (1 - mu - x)
    )
    deriv[3] = -2 * vx + y - ((1 - mu) / r1 ** 3) * y - (mu / r2 ** 3) * y

    return deriv


# ------------------------------------------------------------------------------------------------ #
# deifne RHS function in CR3BP including its state-transition-matrix (Numba compatible)
@jit(nopython=True)
def rhs_cr3bp_with_STM(t, state, mu):
    """Equation of motion in CR3BP along with its STM, compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # coefficients of A matrix
    # first row
    a00 = 0
    a01 = 0
    a02 = 0
    a03 = 1
    a04 = 0
    a05 = 0
    # second row
    a10 = 0
    a11 = 0
    a12 = 0
    a13 = 0
    a14 = 1
    a15 = 0
    # third row
    a20 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a24 = 0
    a25 = 1
    # fourth row
    a33 = 0
    a34 = 2
    a35 = 0
    # fith row
    a43 = -2
    a44 = 0
    a45 = 0
    # sixth row
    a53 = 0
    a54 = 0
    a55 = 0
    # setup vector for dX/dt
    deriv = np.zeros((42,))
    # STATE
    # position derivatives
    deriv[0] = state[3]
    deriv[1] = state[4]
    deriv[2] = state[5]
    # velocitstate derivatives
    deriv[3] = (
        2 * state[4]
        + state[0]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (mu + state[0])
        + (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (1 - mu - state[0])
    )
    deriv[4] = (
        -2 * state[3]
        + state[1]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
    )
    deriv[5] = (
        -((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
    )

    # STATE-TRANSITION MATRIX
    # first row ...
    deriv[6] = (
        a00 * state[6]
        + a01 * state[12]
        + a02 * state[18]
        + a03 * state[24]
        + a04 * state[30]
        + a05 * state[36]
    )
    deriv[7] = (
        a00 * state[7]
        + a01 * state[13]
        + a02 * state[19]
        + a03 * state[25]
        + a04 * state[31]
        + a05 * state[37]
    )
    deriv[8] = (
        a00 * state[8]
        + a01 * state[14]
        + a02 * state[20]
        + a03 * state[26]
        + a04 * state[32]
        + a05 * state[38]
    )
    deriv[9] = (
        a00 * state[9]
        + a01 * state[15]
        + a02 * state[21]
        + a03 * state[27]
        + a04 * state[33]
        + a05 * state[39]
    )
    deriv[10] = (
        a00 * state[10]
        + a01 * state[16]
        + a02 * state[22]
        + a03 * state[28]
        + a04 * state[34]
        + a05 * state[40]
    )
    deriv[11] = (
        a00 * state[11]
        + a01 * state[17]
        + a02 * state[23]
        + a03 * state[29]
        + a04 * state[35]
        + a05 * state[41]
    )

    # second row ...
    deriv[12] = (
        a10 * state[6]
        + a11 * state[12]
        + a12 * state[18]
        + a13 * state[24]
        + a14 * state[30]
        + a15 * state[36]
    )
    deriv[13] = (
        a10 * state[7]
        + a11 * state[13]
        + a12 * state[19]
        + a13 * state[25]
        + a14 * state[31]
        + a15 * state[37]
    )
    deriv[14] = (
        a10 * state[8]
        + a11 * state[14]
        + a12 * state[20]
        + a13 * state[26]
        + a14 * state[32]
        + a15 * state[38]
    )
    deriv[15] = (
        a10 * state[9]
        + a11 * state[15]
        + a12 * state[21]
        + a13 * state[27]
        + a14 * state[33]
        + a15 * state[39]
    )
    deriv[16] = (
        a10 * state[10]
        + a11 * state[16]
        + a12 * state[22]
        + a13 * state[28]
        + a14 * state[34]
        + a15 * state[40]
    )
    deriv[17] = (
        a10 * state[11]
        + a11 * state[17]
        + a12 * state[23]
        + a13 * state[29]
        + a14 * state[35]
        + a15 * state[41]
    )

    # third row ...
    deriv[18] = (
        a20 * state[6]
        + a21 * state[12]
        + a22 * state[18]
        + a23 * state[24]
        + a24 * state[30]
        + a25 * state[36]
    )
    deriv[19] = (
        a20 * state[7]
        + a21 * state[13]
        + a22 * state[19]
        + a23 * state[25]
        + a24 * state[31]
        + a25 * state[37]
    )
    deriv[20] = (
        a20 * state[8]
        + a21 * state[14]
        + a22 * state[20]
        + a23 * state[26]
        + a24 * state[32]
        + a25 * state[38]
    )
    deriv[21] = (
        a20 * state[9]
        + a21 * state[15]
        + a22 * state[21]
        + a23 * state[27]
        + a24 * state[33]
        + a25 * state[39]
    )
    deriv[22] = (
        a20 * state[10]
        + a21 * state[16]
        + a22 * state[22]
        + a23 * state[28]
        + a24 * state[34]
        + a25 * state[40]
    )
    deriv[23] = (
        a20 * state[11]
        + a21 * state[17]
        + a22 * state[23]
        + a23 * state[29]
        + a24 * state[35]
        + a25 * state[41]
    )

    # fourth row ...
    deriv[24] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a33 * state[24]
        + a34 * state[30]
        + a35 * state[36]
    )

    deriv[25] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a33 * state[25]
        + a34 * state[31]
        + a35 * state[37]
    )

    deriv[26] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a33 * state[26]
        + a34 * state[32]
        + a35 * state[38]
    )

    deriv[27] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a33 * state[27]
        + a34 * state[33]
        + a35 * state[39]
    )

    deriv[28] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a33 * state[28]
        + a34 * state[34]
        + a35 * state[40]
    )

    deriv[29] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a33 * state[29]
        + a34 * state[35]
        + a35 * state[41]
    )

    # fifth row ...
    deriv[30] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a43 * state[24]
        + a44 * state[30]
        + a45 * state[36]
    )

    deriv[31] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a43 * state[25]
        + a44 * state[31]
        + a45 * state[37]
    )

    deriv[32] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a43 * state[26]
        + a44 * state[32]
        + a45 * state[38]
    )

    deriv[33] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a43 * state[27]
        + a44 * state[33]
        + a45 * state[39]
    )

    deriv[34] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a43 * state[28]
        + a44 * state[34]
        + a45 * state[40]
    )

    deriv[35] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a43 * state[29]
        + a44 * state[35]
        + a45 * state[41]
    )

    # sixth row ...
    deriv[36] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a53 * state[24]
        + a54 * state[30]
        + a55 * state[36]
    )

    deriv[37] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a53 * state[25]
        + a54 * state[31]
        + a55 * state[37]
    )

    deriv[38] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a53 * state[26]
        + a54 * state[32]
        + a55 * state[38]
    )

    deriv[39] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a53 * state[27]
        + a54 * state[33]
        + a55 * state[39]
    )

    deriv[40] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a53 * state[28]
        + a54 * state[34]
        + a55 * state[40]
    )

    deriv[41] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a53 * state[29]
        + a54 * state[35]
        + a55 * state[41]
    )

    return deriv


# ---------------------------------------------------------------------------------------------------- #
# deifne RHS function in planar CR3BP including its state-transition-matrix (Numba compatible)
@jit(nopython=True)
def rhs_pcr3bp_with_STM(t, state, mu):
    """Equation of motion in planar CR3BP along with its STM, compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 4
        mu (float): CR3BP parameter
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # coefficients of A matrix
    # first row
    a00 = 0
    a01 = 0
    a02 = 0
    a03 = 1
    a04 = 0
    a05 = 0
    # second row
    a10 = 0
    a11 = 0
    a12 = 0
    a13 = 0
    a14 = 1
    a15 = 0
    # third row
    a20 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a24 = 0
    a25 = 1
    # fourth row
    a33 = 0
    a34 = 2
    a35 = 0
    # fith row
    a43 = -2
    a44 = 0
    a45 = 0
    # sixth row
    a53 = 0
    a54 = 0
    a55 = 0
    # setup vector for dX/dt
    deriv = np.zeros((16,))
    # STATE
    # position derivatives
    deriv[0] = state[3]
    deriv[1] = state[4]
    deriv[2] = state[5]
    # velocitstate derivatives
    deriv[3] = (
        2 * state[4]
        + state[0]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (mu + state[0])
        + (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (1 - mu - state[0])
    )
    deriv[4] = (
        -2 * state[3]
        + state[1]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
    )
    deriv[5] = (
        -((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
    )

    # STATE-TRANSITION MATRIX
    # first row ...
    deriv[6] = (
        a00 * state[6]
        + a01 * state[12]
        + a02 * state[18]
        + a03 * state[24]
        + a04 * state[30]
        + a05 * state[36]
    )
    deriv[7] = (
        a00 * state[7]
        + a01 * state[13]
        + a02 * state[19]
        + a03 * state[25]
        + a04 * state[31]
        + a05 * state[37]
    )
    deriv[8] = (
        a00 * state[8]
        + a01 * state[14]
        + a02 * state[20]
        + a03 * state[26]
        + a04 * state[32]
        + a05 * state[38]
    )
    deriv[9] = (
        a00 * state[9]
        + a01 * state[15]
        + a02 * state[21]
        + a03 * state[27]
        + a04 * state[33]
        + a05 * state[39]
    )
    deriv[10] = (
        a00 * state[10]
        + a01 * state[16]
        + a02 * state[22]
        + a03 * state[28]
        + a04 * state[34]
        + a05 * state[40]
    )
    deriv[11] = (
        a00 * state[11]
        + a01 * state[17]
        + a02 * state[23]
        + a03 * state[29]
        + a04 * state[35]
        + a05 * state[41]
    )

    # second row ...
    deriv[12] = (
        a10 * state[6]
        + a11 * state[12]
        + a12 * state[18]
        + a13 * state[24]
        + a14 * state[30]
        + a15 * state[36]
    )
    deriv[13] = (
        a10 * state[7]
        + a11 * state[13]
        + a12 * state[19]
        + a13 * state[25]
        + a14 * state[31]
        + a15 * state[37]
    )
    deriv[14] = (
        a10 * state[8]
        + a11 * state[14]
        + a12 * state[20]
        + a13 * state[26]
        + a14 * state[32]
        + a15 * state[38]
    )
    deriv[15] = (
        a10 * state[9]
        + a11 * state[15]
        + a12 * state[21]
        + a13 * state[27]
        + a14 * state[33]
        + a15 * state[39]
    )
    deriv[16] = (
        a10 * state[10]
        + a11 * state[16]
        + a12 * state[22]
        + a13 * state[28]
        + a14 * state[34]
        + a15 * state[40]
    )
    deriv[17] = (
        a10 * state[11]
        + a11 * state[17]
        + a12 * state[23]
        + a13 * state[29]
        + a14 * state[35]
        + a15 * state[41]
    )

    # third row ...
    deriv[18] = (
        a20 * state[6]
        + a21 * state[12]
        + a22 * state[18]
        + a23 * state[24]
        + a24 * state[30]
        + a25 * state[36]
    )
    deriv[19] = (
        a20 * state[7]
        + a21 * state[13]
        + a22 * state[19]
        + a23 * state[25]
        + a24 * state[31]
        + a25 * state[37]
    )
    deriv[20] = (
        a20 * state[8]
        + a21 * state[14]
        + a22 * state[20]
        + a23 * state[26]
        + a24 * state[32]
        + a25 * state[38]
    )
    deriv[21] = (
        a20 * state[9]
        + a21 * state[15]
        + a22 * state[21]
        + a23 * state[27]
        + a24 * state[33]
        + a25 * state[39]
    )
    deriv[22] = (
        a20 * state[10]
        + a21 * state[16]
        + a22 * state[22]
        + a23 * state[28]
        + a24 * state[34]
        + a25 * state[40]
    )
    deriv[23] = (
        a20 * state[11]
        + a21 * state[17]
        + a22 * state[23]
        + a23 * state[29]
        + a24 * state[35]
        + a25 * state[41]
    )

    # fourth row ...
    deriv[24] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a33 * state[24]
        + a34 * state[30]
        + a35 * state[36]
    )

    deriv[25] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a33 * state[25]
        + a34 * state[31]
        + a35 * state[37]
    )

    deriv[26] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a33 * state[26]
        + a34 * state[32]
        + a35 * state[38]
    )

    deriv[27] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a33 * state[27]
        + a34 * state[33]
        + a35 * state[39]
    )

    deriv[28] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a33 * state[28]
        + a34 * state[34]
        + a35 * state[40]
    )

    deriv[29] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a33 * state[29]
        + a34 * state[35]
        + a35 * state[41]
    )

    # fifth row ...
    deriv[30] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a43 * state[24]
        + a44 * state[30]
        + a45 * state[36]
    )

    deriv[31] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a43 * state[25]
        + a44 * state[31]
        + a45 * state[37]
    )

    deriv[32] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a43 * state[26]
        + a44 * state[32]
        + a45 * state[38]
    )

    deriv[33] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a43 * state[27]
        + a44 * state[33]
        + a45 * state[39]
    )

    deriv[34] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a43 * state[28]
        + a44 * state[34]
        + a45 * state[40]
    )

    deriv[35] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a43 * state[29]
        + a44 * state[35]
        + a45 * state[41]
    )

    # sixth row ...
    deriv[36] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[6]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[12]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[18]
        + a53 * state[24]
        + a54 * state[30]
        + a55 * state[36]
    )

    deriv[37] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[7]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[13]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[19]
        + a53 * state[25]
        + a54 * state[31]
        + a55 * state[37]
    )

    deriv[38] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[8]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[14]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[20]
        + a53 * state[26]
        + a54 * state[32]
        + a55 * state[38]
    )

    deriv[39] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[9]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[15]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[21]
        + a53 * state[27]
        + a54 * state[33]
        + a55 * state[39]
    )

    deriv[40] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[10]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[16]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[22]
        + a53 * state[28]
        + a54 * state[34]
        + a55 * state[40]
    )

    deriv[41] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[11]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[17]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
        )
        * state[23]
        + a53 * state[29]
        + a54 * state[35]
        + a55 * state[41]
    )

    return deriv


# ------------------------------------------------------------------------------------------------ #
# right-hand side function in BCR4BP, centered at planet-moon barycenter
@jit(nopython=True)
def rhs_bcr4bp(t, state, mu, mu3, a, t0, omega_s, eps):
    """Equation of motion in BCR4BP, centered around planet-moon barycenter

    Args:
        t (float): time
        state (np.array): state-vector, length 6 (x,y,z,vx,vy,vz)
        mu (float): CR3BP parameter of planet-moon systems
        mu3 (float): mass of sun, scaled by mass of planet+moon
        a (float): sun-planet+moon barycenter distance, scaled by planet-moon distance
        t0 (float): initial time
        omega_s (float): synodic rotation rate
        eps (float): attenuation factor of sun's gravitational effect, 0~1

    Returns:
        (np.array): derivative of state
    """

    # unpack state
    x = state[0]
    y = state[1]
    z = state[2]
    vx = state[3]
    vy = state[4]
    vz = state[5]

    # compute radii to planet (r1), moon (r2)
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)

    # compute sun position and radius (r3)
    xs = a * np.cos(omega_s * t + t0)
    ys = a * np.sin(omega_s * t + t0)
    zs = 0.0
    r3 = np.sqrt((x - xs) ** 2 + (y - ys) ** 2 + (z - zs) ** 2)

    # compute derivative of state
    deriv = np.zeros(
        6,
    )
    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] = (
        2 * vy
        + x
        - ((1 - mu) / r1 ** 3) * (mu + x)
        + (mu / r2 ** 3) * (1 - mu - x)
        + eps * (-(mu3 / r3 ** 3) * (x - xs) - (mu3 / a ** 3) * xs)
    )
    deriv[4] = (
        -2 * vx
        + y
        - ((1 - mu) / r1 ** 3) * y
        - (mu / r2 ** 3) * y
        + eps * (-(mu3 / r3 ** 3) * (y - ys) - (mu3 / a ** 3) * ys)
    )
    deriv[5] = (
        -((1 - mu) / r1 ** 3) * z
        - (mu / r2 ** 3) * z
        + eps * (-(mu3 / r3 ** 3) * (z) - (mu3 / a ** 3) * zs)
    )

    return deriv


# ------------------------------------------------------------------------------------------------ #
# right-hand side function in BCR4BP with STM, centered at cenetred at planet-moon barycenter
@jit(nopython=True)
def rhs_bcr4bp_with_STM(t, state, mu, mu3, a, t0, omega_s, eps):
    """Equation of motion in BCR4BP along with its STM, compatible with njit

    Args:
        t (float): time
        state (np.array): state-vector, length 6 (x,y,z,vx,vy,vz)
        mu (float): CR3BP parameter of planet-moon systems
        mu3 (float): mass of sun, scaled by mass of planet+moon
        a (float): sun-planet+moon barycenter distance, scaled by planet-moon distance
        t0 (float): initial time
        omega_s (float): synodic rotation rate
        eps (float): attenuation factor of sun's gravitational effect, 0~1

    Returns:
        (np.array): 1D array of derivative of Cartesian state and STM
    """
    # coefficients of A matrix
    # first row
    a00 = 0
    a01 = 0
    a02 = 0
    a03 = 1
    a04 = 0
    a05 = 0
    # second row
    a10 = 0
    a11 = 0
    a12 = 0
    a13 = 0
    a14 = 1
    a15 = 0
    # third row
    a20 = 0
    a21 = 0
    a22 = 0
    a23 = 0
    a24 = 0
    a25 = 1
    # fourth row
    a33 = 0
    a34 = 2
    a35 = 0
    # fith row
    a43 = -2
    a44 = 0
    a45 = 0
    # sixth row
    a53 = 0
    a54 = 0
    a55 = 0
    # setup vector for dX/dt
    deriv = np.zeros((42,))
    # STATE
    # position derivatives
    deriv[0] = state[3]  # vx
    deriv[1] = state[4]  # vy
    deriv[2] = state[5]  # vz
    # associated quantities
    # xs: a*np.cos(omega_s*t + t0)
    # ys: a*np.sin(omega_s*t + t0)
    # zs: 0.0
    # r1: ((state[0]+mu)**2 + state[1]**2 + state[2]**2)**1.5
    # r2: ((state[0]-1+mu)**2 + state[1]**2 + state[2]**2)**1.5
    # r3: ((state[0]-a*np.cos(omega_s*t + t0))**2 + (state[1]-a*np.sin(omega_s*t + t0))**2 + (state[2])**2 )**1.5
    # velocity derivatives
    deriv[3] = (
        2 * state[4]
        + state[0]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (mu + state[0])
        + (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * (1 - mu - state[0])
        + eps
        * (
            -(
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 1.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0))
            - (mu3 / a ** 3) * a * np.cos(omega_s * t + t0)
        )
    )
    deriv[4] = (
        -2 * state[3]
        + state[1]
        - ((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[1]
        + eps
        * (
            -(
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 1.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            - (mu3 / a ** 3) * a * np.sin(omega_s * t + t0)
        )
    )
    deriv[5] = (
        -((1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
        - (mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5)
        * state[2]
        + eps
        * (
            -(
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 1.5
            )
            * (state[2])
        )
    )

    # STATE-TRANSITION MATRIX
    # first row ...
    deriv[6] = (
        a00 * state[6]
        + a01 * state[12]
        + a02 * state[18]
        + a03 * state[24]
        + a04 * state[30]
        + a05 * state[36]
    )
    deriv[7] = (
        a00 * state[7]
        + a01 * state[13]
        + a02 * state[19]
        + a03 * state[25]
        + a04 * state[31]
        + a05 * state[37]
    )
    deriv[8] = (
        a00 * state[8]
        + a01 * state[14]
        + a02 * state[20]
        + a03 * state[26]
        + a04 * state[32]
        + a05 * state[38]
    )
    deriv[9] = (
        a00 * state[9]
        + a01 * state[15]
        + a02 * state[21]
        + a03 * state[27]
        + a04 * state[33]
        + a05 * state[39]
    )
    deriv[10] = (
        a00 * state[10]
        + a01 * state[16]
        + a02 * state[22]
        + a03 * state[28]
        + a04 * state[34]
        + a05 * state[40]
    )
    deriv[11] = (
        a00 * state[11]
        + a01 * state[17]
        + a02 * state[23]
        + a03 * state[29]
        + a04 * state[35]
        + a05 * state[41]
    )

    # second row ...
    deriv[12] = (
        a10 * state[6]
        + a11 * state[12]
        + a12 * state[18]
        + a13 * state[24]
        + a14 * state[30]
        + a15 * state[36]
    )
    deriv[13] = (
        a10 * state[7]
        + a11 * state[13]
        + a12 * state[19]
        + a13 * state[25]
        + a14 * state[31]
        + a15 * state[37]
    )
    deriv[14] = (
        a10 * state[8]
        + a11 * state[14]
        + a12 * state[20]
        + a13 * state[26]
        + a14 * state[32]
        + a15 * state[38]
    )
    deriv[15] = (
        a10 * state[9]
        + a11 * state[15]
        + a12 * state[21]
        + a13 * state[27]
        + a14 * state[33]
        + a15 * state[39]
    )
    deriv[16] = (
        a10 * state[10]
        + a11 * state[16]
        + a12 * state[22]
        + a13 * state[28]
        + a14 * state[34]
        + a15 * state[40]
    )
    deriv[17] = (
        a10 * state[11]
        + a11 * state[17]
        + a12 * state[23]
        + a13 * state[29]
        + a14 * state[35]
        + a15 * state[41]
    )

    # third row ...
    deriv[18] = (
        a20 * state[6]
        + a21 * state[12]
        + a22 * state[18]
        + a23 * state[24]
        + a24 * state[30]
        + a25 * state[36]
    )
    deriv[19] = (
        a20 * state[7]
        + a21 * state[13]
        + a22 * state[19]
        + a23 * state[25]
        + a24 * state[31]
        + a25 * state[37]
    )
    deriv[20] = (
        a20 * state[8]
        + a21 * state[14]
        + a22 * state[20]
        + a23 * state[26]
        + a24 * state[32]
        + a25 * state[38]
    )
    deriv[21] = (
        a20 * state[9]
        + a21 * state[15]
        + a22 * state[21]
        + a23 * state[27]
        + a24 * state[33]
        + a25 * state[39]
    )
    deriv[22] = (
        a20 * state[10]
        + a21 * state[16]
        + a22 * state[22]
        + a23 * state[28]
        + a24 * state[34]
        + a25 * state[40]
    )
    deriv[23] = (
        a20 * state[11]
        + a21 * state[17]
        + a22 * state[23]
        + a23 * state[29]
        + a24 * state[35]
        + a25 * state[41]
    )

    # fourth row ... (Uxx, Uxy, Uxz)
    deriv[24] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[6]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[12]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[18]
        + a33 * state[24]
        + a34 * state[30]
        + a35 * state[36]
    )

    deriv[25] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[7]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[13]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[19]
        + a33 * state[25]
        + a34 * state[31]
        + a35 * state[37]
    )

    deriv[26] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[8]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[14]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[20]
        + a33 * state[26]
        + a34 * state[32]
        + a35 * state[38]
    )

    deriv[27] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[9]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[15]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[21]
        + a33 * state[27]
        + a34 * state[33]
        + a35 * state[39]
    )

    deriv[28] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[10]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[16]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[22]
        + a33 * state[28]
        + a34 * state[34]
        + a35 * state[40]
    )

    deriv[29] = (
        (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * (
                (state[0] + mu) ** 2
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1) ** 2
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[0] - a * np.cos(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[11]
        + (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[17]
        + (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[23]
        + a33 * state[29]
        + a34 * state[35]
        + a35 * state[41]
    )

    # fifth row ... (Uyx, Uyy, Uzz)
    deriv[30] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[6]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[12]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[18]
        + a43 * state[24]
        + a44 * state[30]
        + a45 * state[36]
    )

    deriv[31] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[7]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[13]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[19]
        + a43 * state[25]
        + a44 * state[31]
        + a45 * state[37]
    )

    deriv[32] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[8]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[14]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[20]
        + a43 * state[26]
        + a44 * state[32]
        + a45 * state[38]
    )

    deriv[33] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[9]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[15]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[21]
        + a43 * state[27]
        + a44 * state[33]
        + a45 * state[39]
    )

    deriv[34] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[10]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[16]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[22]
        + a43 * state[28]
        + a44 * state[34]
        + a45 * state[40]
    )

    deriv[35] = (
        (
            3
            * state[1]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0))
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[11]
        + (
            1
            - (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[1] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[1] - a * np.sin(omega_s * t + t0)) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[17]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[23]
        + a43 * state[29]
        + a44 * state[35]
        + a45 * state[41]
    )

    # sixth row ... (Uzx, Uzy, Uzz)
    deriv[36] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[6]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[12]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[18]
        + a53 * state[24]
        + a54 * state[30]
        + a55 * state[36]
    )

    deriv[37] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[7]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[13]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[19]
        + a53 * state[25]
        + a54 * state[31]
        + a55 * state[37]
    )

    deriv[38] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[8]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[14]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[20]
        + a53 * state[26]
        + a54 * state[32]
        + a55 * state[38]
    )

    deriv[39] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[9]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[15]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[21]
        + a53 * state[27]
        + a54 * state[33]
        + a55 * state[39]
    )

    deriv[40] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[10]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[16]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[22]
        + a53 * state[28]
        + a54 * state[34]
        + a55 * state[40]
    )

    deriv[41] = (
        (
            3
            * state[2]
            * (
                (state[0] + mu)
                * (1 - mu)
                / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + (state[0] + mu - 1)
                * mu
                / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[0] - a * np.cos(omega_s * t + t0))
        )
        * state[11]
        + (
            3
            * state[1]
            * state[2]
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2])
            * (state[1] - a * np.sin(omega_s * t + t0))
        )
        * state[17]
        + (
            -(1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            - mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
            + 3
            * state[2] ** 2
            * (
                (1 - mu) / ((state[0] + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
                + mu / ((state[0] - 1 + mu) ** 2 + state[1] ** 2 + state[2] ** 2) ** 2.5
            )
            + 3
            * (
                mu3
                / (
                    (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                    + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                    + (state[2]) ** 2
                )
                ** 2.5
            )
            * (state[2]) ** 2
            - mu3
            / (
                (state[0] - a * np.cos(omega_s * t + t0)) ** 2
                + (state[1] - a * np.sin(omega_s * t + t0)) ** 2
                + (state[2]) ** 2
            )
            ** 1.5
        )
        * state[23]
        + a53 * state[29]
        + a54 * state[35]
        + a55 * state[41]
    )

    return deriv


# ------------------------------------------------------------------------------------------------ #
# define RHS function in CCR4BP
@jit(nopython=True)
def rhs_ccr4bp(t, state, mu1, mu2, mu3, om2, om3, theta02, theta03, d2, d3):
    """Right-hand side equations of motion for concentric-circular restricted four body problem
    NOTE this is centered at m1! Transitioning from CR3BP requires shift along x-axis!

    Args:
        t (float):
        state (np.array):
        mu1 (float): mass parameter of first primary
        mu2 (float): mass parameter of second primary
        mu3 (float): mass parameter of third primary
        om2 (float):
        om3 (float):
        theta02 (float):
        theta03 (float):
        d2 (float):
        d3 (float):

    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """
    # unpack state
    x, y, z = state[0], state[1], state[2]
    vx, vy, vz = state[3], state[4], state[5]
    # get location of m2, m3
    theta2 = theta02 + om2 * (t)
    theta3 = theta03 + om3 * (t)
    # position vectors from m1 to m2, m3
    r12 = np.array([1.0, 0.0, 0.0])
    r13 = (d3 / d2) * np.array([np.cos(theta3 - theta2), np.sin(theta3 - theta2), 0.0])
    # locations of spacecraft with respect on m1, m2, m3
    r1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1) ** 2 + y ** 2 + z ** 2)
    r3 = np.sqrt(
        (x - (d3 / d2) * np.cos(theta3 - theta2)) ** 2
        + (y - (d3 / d2) * np.sin(theta3 - theta2)) ** 2
        + z ** 2
    )
    deriv = np.zeros(
        6,
    )
    # position derivatives
    deriv[0], deriv[1], deriv[2] = state[3], state[4], state[5]
    # velocity derivatives
    deriv[3] = (
        2 * vy
        + x
        - mu1 * x / r1 ** 3
        - mu2 * ((x - 1) / r2 ** 3 + 1 / la.norm(r12) ** 3)
        - mu3
        * (
            (x - (d3 / d2) * np.cos(theta3 - theta2)) / r3 ** 3
            + (d3 / d2) * np.cos(theta3 - theta2) / la.norm(r13) ** 3
        )
    )
    deriv[4] = (
        -2 * vx
        + y
        - mu1 * y / r1 ** 3
        - mu2 * y / r2 ** 3
        - mu3
        * (
            (y - (d3 / d2) * np.sin(theta3 - theta2)) / r3 ** 3
            + (d3 / d2) * np.sin(theta3 - theta2) / la.norm(r13) ** 3
        )
    )
    deriv[5] = -mu1 * z / r1 ** 3 - mu2 * z / r2 ** 3 - mu3 * z / r3 ** 3
    return deriv


# ------------------------------------------------------------------------------------------------ #
# define RHS function in CR3BP with constant thrust along direction of motion
@jit(nopython=True)
def rhs_cr3bp_constantthrust(t, state, mu, a_thrust):
    """Equation of motion in CR3BP, formulated for scipy.integrate.solve=ivp(), compatible with njit

    Args:
        t (float): time
        state (np.array): 1D array of Cartesian state, length 6
        mu (float): CR3BP parameter
        a_thrust (float): thrust-based acceleration
    Returns:
        (np.array): 1D array of derivative of Cartesian state
    """

    # unpack positions
    x = state[0]
    y = state[1]
    z = state[2]
    # unpack velocities
    vx = state[3]
    vy = state[4]
    vz = state[5]
    # compute radii to each primary
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)
    # setup vector for dX/dt
    deriv = np.zeros((6,))
    # compute accelertaion in each direction
    a_thrust_x = a_thrust * vx / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    a_thrust_y = a_thrust * vx / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    a_thrust_z = a_thrust * vx / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # position derivatives
    deriv[0] = vx
    deriv[1] = vy
    deriv[2] = vz
    # velocity derivatives
    deriv[3] = (
        2 * vy
        + x
        - ((1 - mu) / r1 ** 3) * (mu + x)
        + (mu / r2 ** 3) * (1 - mu - x)
        + a_thrust_x
    )
    deriv[4] = -2 * vx + y - ((1 - mu) / r1 ** 3) * y - (mu / r2 ** 3) * y + a_thrust_y
    deriv[5] = -((1 - mu) / r1 ** 3) * z - (mu / r2 ** 3) * z + a_thrust_z

    return deriv
