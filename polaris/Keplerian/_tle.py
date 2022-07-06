"""
Functions for handling TLE
Workflow:
    - get_tle() saves TLE data as .txt files from celestrak
    - tle_list_to_df() takes the file pathj and constructs a pandas.DataFrame, calling tle2dict() at each TLE
    - filter_tle_dataframe() filters a TLE DataFrame based on user-specified condition
    - tledf_to_spreadsheet() constructs a spreadsheet sheet from TLE dataframe
"""

import numpy as np
import pandas as pd
import os

try:
    from ._orbitalElements import meanAnom2trueAnom
    from ._statevec2elements import print_elts
except:
    from _orbitalElements import meanAnom2trueAnom
    from _statevec2elements import print_elts


def tle_list_to_df(tlefile, sat_type=None, mu=398600.44, tol=1.e-12):
    """Convert .txt file with multiple tles to pandas a DataFrame
    Args:
        tlefile (str): path to tle file
        sat_type (str): type of satellite family
    Returns:
        (pd.DataFrame)
    """
    # read text file into list
    with open(tlefile) as f:
        tlefles = f.readlines()
        f.close()
    # separate list in sets of three
    sats = []
    for idxTle in range(int(len(tlefles)/3)):
        idx_start = idxTle*3
        idx_end   = idxTle*3+3
        sats.append( tlefles[idx_start:idx_end ] )
    # convert into dictionary, then to DataFrames
    dfsat = pd.DataFrame(columns=[el for el in tle2dict(sats[0], mu, tol).keys()])
    count = 0
    for idx, sat in enumerate(sats):
        snew = pd.Series(tle2dict(sats[idx], mu, tol).values(), index=[el for el in tle2dict(sats[idx], mu, tol).keys()], name=int(idx))
        dfsat = pd.DataFrame.append(dfsat , snew)
    # assign satellite type if given
    if sat_type is not None:
        dfsat.assign( sat_type = [sat_type]*len(dfsat))
    return dfsat


def print_tle(tlefile, sat_type=None, mu=398600.44, tol=1.e-12):
    dfsat = tle_list_to_df(tlefile, sat_type, mu, tol)
    print_elts(dfsat.iloc[0,:].to_dict())
    return


def tle2dict(tle, gm_earth=398600.44, tol=1.e-11):
    """TLE file as list of text, per line"""
    deg2rad = np.pi/180
    line0, line1, line2 = tle[0], tle[1], tle[2]
    tle_dict = {
        # satellite name
        "satellite_name": line0.split('  ')[0], 
        # first line
        "satellite_number":  int(   line1[2:7]   ), 
        "classification":    line1[7],
        "internationalDesignator": line1[9:17],
        "epochYear":         int( line1[18:20] ),
        "epoch":             float( line1[20:32] ),
        "dMeanMotionDt":     line1[33:43],
        "d2MeanMotionDt2":   line1[44:52],
        "BSTARdrag":         line1[53:61],
        "ephemerisType":     line1[62],
        "elementNumber":     int( line1[64:68] ),
        # second line 
        "inc":               float( line2[8:16]  )*deg2rad,
        "raan":              float( line2[17:25] )*deg2rad,
        "ecc":               float( "0." + line2[26:33] ),#/10**(len(line2[26:33])),
        "aop":               float( line2[34:42] )*deg2rad,
        "meanAnom":          float( line2[43:51] )*deg2rad,
        "meanMotion":        float( line2[52:63] ),
        "revNumberAtEpoch":  float( line2[63:68] ),
    }
    # additionally compute Keplerian elements
    n = float( line2[52:63] )  # mean motion, rev/day
    #print(f"Period: {86400/n /60} min")
    tle_dict["period"] = 86400/n  # second/rev
    tle_dict["sma"] = (gm_earth/(2*np.pi*n/86400)**2)**(1/3)
    # gm_earth**(1/3) * (tle_dict["period"]/(2*np.pi))**(2/3)
    #print( float( line2[43:51] )*deg2rad )
    tle_dict["ta"]  = meanAnom2trueAnom(meanAnom=float( line2[43:51] )*deg2rad, ecc= float( line2[26:33] )/10**(len(line2[26:33])), tol=tol)
    return tle_dict