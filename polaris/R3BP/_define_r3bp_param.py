#!/usr/bin/env python3
"""
Functions define R3BP parameters
Yuri Shimane
"""

import numpy as np

from .. import SolarSystemConstants as sscs
from .. import Keplerian as kepl


class CR3BP:
    def __init__(self, m1_naifID, m2_naifID):
        """CR3BP parameter class

        Args:
            m1_naifID (int): NAIF ID of first body
            m2_naifID (int): NAIF ID of second body
        """
        # list of semi-major axis
        if m1_naifID == 10:
            self.a1 = 0.0
            self.a2 = sscs.get_semiMajorAxes(m2_naifID)[0]
        elif m2_naifID == 10:
            self.a1 = 0.0
            self.a2 = sscs.get_semiMajorAxes(m1_naifID)[0]
        else:
            semiMajorAxes = sscs.get_semiMajorAxes(m1_naifID, m2_naifID)
            self.a1, self.a2 = semiMajorAxes[0], semiMajorAxes[1]
        # list of gm
        gmlst = sscs.get_gm(m1_naifID, m2_naifID)
        self.m1_gm, self.m2_gm = gmlst[0], gmlst[1]
        if self.m1_gm < self.m2_gm:
            raise Exception("Excepting m1 > m2!")
        # create list of parameters
        self.mu = self.m2_gm / (self.m1_gm + self.m2_gm)
        self.lstar = self.a2
        self.tstar = np.sqrt((self.a2) ** 3 / (self.m1_gm + self.m2_gm))
        self.vstar = self.lstar / self.tstar
        self.mstar = self.m1_gm + self.m2_gm
        self.m2_soi = self.a2 * (self.m2_gm / self.m1_gm) ** (2 / 5)


class BCR4BP:
    def __init__(self, m1_naifID, m2_naifID):
        """CR3BP parameter class

        Args:
            m1_naifID (int): NAIF ID of first body (excluding Sun)
            m2_naifID (int): NAIF ID of second body (excluding Sun)
        """
        cr3bp = CR3BP(m1_naifID, m2_naifID)

        # get information of Sun
        gm_sun = sscs.get_gm(10)[0]
        a_m1_sun = sscs.get_semiMajorAxes(m1_naifID)[
            0
        ]  # semi-major axis of first (larger) body
        period_sun = 2 * np.pi * np.sqrt(a_m1_sun ** 3 / gm_sun)  # [sec]

        # compute rotation rate of sun about m1-m2 barycenter
        tsyn = kepl.get_synodic_period(period_sun, 2 * np.pi * cr3bp.tstar)  # [sec]
        om_sun = -2 * np.pi / (tsyn / cr3bp.tstar)  # [rad/canonical time]

        # compute scaled gm and length of sun
        a_sun = a_m1_sun / cr3bp.lstar
        mu3 = gm_sun / cr3bp.mstar

        # house and return parameter object
        self.om_sun = om_sun
        self.a_sun = a_sun
        self.mu3 = mu3
        # inherit parameters from cr3bp
        self.mu = cr3bp.mu
        self.lstar = cr3bp.lstar
        self.tstar = cr3bp.tstar
        self.vstar = cr3bp.lstar / cr3bp.tstar
        self.mstar = cr3bp.mstar
        self.m2_soi = cr3bp.m2_soi


def get_bcr4bp_param(m1_naifID, m2_naifID):
    """Function returns bcr4bp parameters mu, Lstar, Tstar, soi of m2,

    Args:
        m1_naifID (str): mass of first primary, m1 > m2
        m2_naifID (str): mass of second primary, m1 > m2

    Returns:
        (obj): object with fields:
            mu (float): mass-parameter
            lstar (float): non-dimensional distance
            tstar (float): non-dimensional time
            m2_soi (float): sphere of influence of second mass, in km
    """
    # initialize empty parameter object
    paramBCR4BP = Parameters()
    # inherit parameters from the CR3BP system
    paramCR3BP = get_cr3bp_param(m1_naifID, m2_naifID)

    # get information of Sun
    gm_sun = sscs.get_gm("10")[0]
    a_m1_sun = sscs.get_semiMajorAxes(m1_naifID)[
        0
    ]  # semi-major axis of first (larger) body
    period_sun = 2 * np.pi * np.sqrt(a_m1_sun ** 3 / gm_sun)  # [sec]

    # compute rotation rate of sun about m1-m2 barycenter
    tsyn = kepl.get_synodic_period(period_sun, 2 * np.pi * paramCR3BP.tstar)  # [sec]
    om_sun = -2 * np.pi / (tsyn / paramCR3BP.tstar)  # [rad/canonical time]

    # compute scaled gm and length of sun
    a_sun = a_m1_sun / paramCR3BP.lstar
    mu3 = gm_sun / paramCR3BP.mstar

    # house and return parameter object
    paramBCR4BP.om_sun = om_sun
    paramBCR4BP.a_sun = a_sun
    paramBCR4BP.mu3 = mu3
    # inherit parameters from cr3bp
    paramBCR4BP.mu = paramCR3BP.mu
    paramBCR4BP.lstar = paramCR3BP.lstar
    paramBCR4BP.tstar = paramCR3BP.tstar
    paramBCR4BP.vstar = paramCR3BP.lstar / paramCR3BP.tstar
    paramBCR4BP.mstar = paramCR3BP.mstar
    paramBCR4BP.m2_soi = paramCR3BP.m2_soi
    return paramBCR4BP


# initialise
class Parameters:
    pass


# ----------------------------------------------------------------------------------------------- #
def get_cr3bp_mu(m1_naifID, m2_naifID):
    """Function returns mass-parameter mu of the CR3BP system

    Args:
        m1_naifID (str): mass of first primary
        m2_naifID (str): mass of second primary
    Returns:
        (float): mass-parameter mu
    """
    # list of gm
    gmlst = sscs.get_gm(m1_naifID, m2_naifID)
    m1_gm, m2_gm = gmlst[0], gmlst[1]
    if m1_gm > m2_gm:
        return m2_gm / (m1_gm + m2_gm)
    else:
        return m1_gm / (m1_gm + m2_gm)


def get_bcr4bp_mus(m1_naifID, m2_naifID):
    """Function returns mass-parameter mu of the CR3BP system

    Args:
        m1_naifID (str): mass of first primary
        m2_naifID (str): mass of second primary
    Returns:
        (float): mass-parameter mu

    Raises:

    """
    # list of gm
    gmlst = sscs.get_gm(m1_naifID, m2_naifID, "10")
    m1_gm, m2_gm, m3_gm = gmlst[0], gmlst[1], gmlst[2]
    if m1_gm < m2_gm:
        raise Exception("Excepting m1 > m2!")
    mu = m2_gm / (m1_gm + m2_gm)
    mu3 = m3_gm / (m1_gm + m2_gm)
    return mu, mu3


# ----------------------------------------------------------------------------------------------- #
def get_cr3bp_param(m1_naifID, m2_naifID):
    """Function returns CR3BP parameters mu, Lstar, Tstar, soi of m2

    Args:
        m1_naifID (str): mass of first primary
        m2_naifID (str): mass of second primary

    Returns:
        (obj): object with fields:
            mu (float): mass-parameter
            lstar (float): non-dimensional distance
            tstar (float): non-dimensional time
            m2_soi (float): sphere of influence of second mass, in km
    """
    # initialize empty parameter object
    paramCR3BP = Parameters()
    # list of semi-major axis
    if m1_naifID == 10:
        a2 = sscs.get_semiMajorAxes(m2_naifID)[0]
    elif m2_naifID == 10:
        a2 = sscs.get_semiMajorAxes(m1_naifID)[0]
    else:
        semiMajorAxes = sscs.get_semiMajorAxes(m1_naifID, m2_naifID)
        a1, a2 = semiMajorAxes[0], semiMajorAxes[1]

    # list of gm
    gmlst = sscs.get_gm(m1_naifID, m2_naifID)
    m1_gm, m2_gm = gmlst[0], gmlst[1]
    if m1_gm < m2_gm:
        raise Exception("Excepting m1 > m2!")
    # create list of parameters
    paramCR3BP.mu = m2_gm / (m1_gm + m2_gm)
    paramCR3BP.lstar = a2
    paramCR3BP.tstar = np.sqrt((a2) ** 3 / (m1_gm + m2_gm))
    paramCR3BP.vstar = paramCR3BP.lstar / paramCR3BP.tstar
    paramCR3BP.mstar = m1_gm + m2_gm
    paramCR3BP.m2_soi = a2 * (m2_gm / m1_gm) ** (2 / 5)
    return paramCR3BP


def get_bcr4bp_param(m1_naifID, m2_naifID):
    """Function returns bcr4bp parameters mu, Lstar, Tstar, soi of m2,

    Args:
        m1_naifID (str): mass of first primary, m1 > m2
        m2_naifID (str): mass of second primary, m1 > m2

    Returns:
        (obj): object with fields:
            mu (float): mass-parameter
            lstar (float): non-dimensional distance
            tstar (float): non-dimensional time
            m2_soi (float): sphere of influence of second mass, in km
    """
    # initialize empty parameter object
    paramBCR4BP = Parameters()
    # inherit parameters from the CR3BP system
    paramCR3BP = get_cr3bp_param(m1_naifID, m2_naifID)

    # get information of Sun
    gm_sun = sscs.get_gm(10)[0]
    a_m1_sun = sscs.get_semiMajorAxes(m1_naifID)[
        0
    ]  # semi-major axis of first (larger) body
    period_sun = 2 * np.pi * np.sqrt(a_m1_sun ** 3 / gm_sun)  # [sec]

    # compute rotation rate of sun about m1-m2 barycenter
    tsyn = kepl.get_synodic_period(period_sun, 2 * np.pi * paramCR3BP.tstar)  # [sec]
    om_sun = -2 * np.pi / (tsyn / paramCR3BP.tstar)  # [rad/canonical time]

    # compute scaled gm and length of sun
    a_sun = a_m1_sun / paramCR3BP.lstar
    mu3 = gm_sun / paramCR3BP.mstar

    # house and return parameter object
    paramBCR4BP.om_sun = om_sun
    paramBCR4BP.a_sun = a_sun
    paramBCR4BP.mu3 = mu3
    # inherit parameters from cr3bp
    paramBCR4BP.mu = paramCR3BP.mu
    paramBCR4BP.lstar = paramCR3BP.lstar
    paramBCR4BP.tstar = paramCR3BP.tstar
    paramBCR4BP.vstar = paramCR3BP.lstar / paramCR3BP.tstar
    paramBCR4BP.mstar = paramCR3BP.mstar
    paramBCR4BP.m2_soi = paramCR3BP.m2_soi
    return paramBCR4BP
