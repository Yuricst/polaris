#!/usr/bin/env python3
"""
Functions define R3BP parameters
"""


from .. import SolarSystemConstants as ssc


def get_cr3bp_mu(m1_naifID, m2_naifID):
    """Function returns mass-parameter mu of the CR3BP system

    Args:
        m1_naifID (str): mass of first primary
        m2_naifID (str): mass of second primary
    Returns:
        (float): mass-parameter mu
    """
    # list of gm
    gmlst = ssc.get_gm(m1_naifID, m2_naifID)
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
    gmlst = ssc.get_gm(m1_naifID, m2_naifID, "10")
    m1_gm, m2_gm, m3_gm = gmlst[0], gmlst[1], gmlist[2]
    if m1_gm < m2_gm:
        raise Exception("Excepting m1 > m2!")
    mu  = m2_gm / (m1_gm + m2_gm)
    mu3 = m3_gm / (m1_gm + m2_gm)
    return mu, mu3



def get_cr3bp_param(m1_naifID, m2_naifID, lstar):
    """Function returns CR3BP parameters mu, Lstar, Tstar, soi of m2
    Args:

    Returns:
        (obj): object with fields:
            mu (float): mass-parameter
            lstar (float): non-dimensional distance
            tstar (float): non-dimensional time
            m2_soi (float): sphere of influence of second mass, in km
    """

    # initialise
    class Parameters():
        pass
    paramCR3BP = Parameters()

    # list of gm
    gmlst = ssc.get_gm(m1_naifID, m2_naifID)
    m1_gm, m2_gm = gmlst[0], gmlst[1]
    if m1_gm < m2_gm:
        raise Exception("Excepting m1 > m2!")
    paramCR3BP.mu     = m2_gm / (m1_gm + m2_gm)
    paramCR3BP.tstar  = np.sqrt( ( lstar )**3 / ( m1_gm + m2_gm ) )
    paramCR3BP.lstar  = ltar
    paramCR3BP.m2_soi = lstar * (m2_gm/m1_gm)**(2/5)
    return paramCR3BP


