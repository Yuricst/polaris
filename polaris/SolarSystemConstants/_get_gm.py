#!/usr/bin/env python3
"""
Functions define GM values of solar system objects
"""


# ------------------------------------------------------------------------- #
def get_gm(*args):
    """Function returns GM value of body specified by NAIF ID. 
    For body names, refer to: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html

    Args:
        naifIDs (str): tuple containing strings of naif ID to use to extract GM values. Multiple naifIDs may be passed in a single function call.

    Returns:
        (lst): lst of GM values

    Examples:
        >>> get_gm("399", "301")
        [398600.435436096, 4902.800066163796]
    """
    # call gm values
    de431gm = get_gm_de431()
    gm_out = []
    for naifID in args:
        bdyname = "BODY" + str(naifID) + "_GM"
        gm_out.append( de431gm[bdyname] )
    return gm_out



# ------------------------------------------------------------------------- #
def get_gm_de431():
    """Function returns tuple of gm values from de431 spice kernel
    
    Args:
        None
    Returns:
        (dict): dictionary with fields defined by "BODY" + <NAIF body ID> + "_GM"", which contains a tuple of the GM value of the corresponding body
    """
    de431 = {
        "BODY1_GM"       : 2.2031780000000021E+04 ,
        "BODY2_GM"       : 3.2485859200000006E+05 ,
        "BODY3_GM"       : 4.0350323550225981E+05 ,
        "BODY4_GM"       : 4.2828375214000022E+04 ,
        "BODY5_GM"       : 1.2671276480000021E+08 ,
        "BODY6_GM"       : 3.7940585200000003E+07 ,
        "BODY7_GM"       : 5.7945486000000080E+06 ,
        "BODY8_GM"       : 6.8365271005800236E+06 ,
        "BODY9_GM"       : 9.7700000000000068E+02 ,
        "BODY10_GM"      : 1.3271244004193938E+11 ,

        "BODY199_GM"     : 2.2031780000000021E+04 ,
        "BODY299_GM"     : 3.2485859200000006E+05 ,
        "BODY399_GM"     : 3.9860043543609598E+05 ,
        "BODY499_GM"     : 4.282837362069909E+04  ,
        "BODY599_GM"     : 1.266865349218008E+08  ,
        "BODY699_GM"     : 3.793120749865224E+07  ,
        "BODY799_GM"     : 5.793951322279009E+06  ,
        "BODY899_GM"     : 6.835099502439672E+06  ,
        "BODY999_GM"     : 8.696138177608748E+02  ,

        "BODY301_GM"     : 4.9028000661637961E+03 ,

        "BODY401_GM"     : 7.087546066894452E-04 ,
        "BODY402_GM"     : 9.615569648120313E-05 ,

        "BODY501_GM"     : 5.959916033410404E+03 ,
        "BODY502_GM"     : 3.202738774922892E+03 ,
        "BODY503_GM"     : 9.887834453334144E+03 ,
        "BODY504_GM"     : 7.179289361397270E+03 ,
        "BODY505_GM"     : 1.378480571202615E-01 ,

        "BODY601_GM"     : 2.503522884661795E+00 ,
        "BODY602_GM"     : 7.211292085479989E+00 ,
        "BODY603_GM"     : 4.121117207701302E+01 ,
        "BODY604_GM"     : 7.311635322923193E+01 ,
        "BODY605_GM"     : 1.539422045545342E+02 ,
        "BODY606_GM"     : 8.978138845307376E+03 ,
        "BODY607_GM"     : 3.718791714191668E-01 ,
        "BODY608_GM"     : 1.205134781724041E+02 ,
        "BODY609_GM"     : 5.531110414633374E-01 ,
        "BODY610_GM"     : 1.266231296945636E-01 ,
        "BODY611_GM"     : 3.513977490568457E-02 ,
        "BODY615_GM"     : 3.759718886965353E-04 ,
        "BODY616_GM"     : 1.066368426666134E-02 ,
        "BODY617_GM"     : 9.103768311054300E-03 ,

        "BODY701_GM"     : 8.346344431770477E+01 ,
        "BODY702_GM"     : 8.509338094489388E+01 ,
        "BODY703_GM"     : 2.269437003741248E+02 ,
        "BODY704_GM"     : 2.053234302535623E+02 ,
        "BODY705_GM"     : 4.319516899232100E+00 ,

        "BODY801_GM"     : 1.427598140725034E+03 ,

        "BODY901_GM"     : 1.058799888601881E+02 ,
        "BODY902_GM"     : 3.048175648169760E-03 ,
        "BODY903_GM"     : 3.211039206155255E-03 ,
        "BODY904_GM"     : 1.110040850536676E-03 ,

        "BODY2000001_GM" : 6.3130000000000003E+01 ,
        "BODY2000002_GM" : 1.3730000000000000E+01 ,
        "BODY2000003_GM" : 1.8200000000000001E+00 ,
        "BODY2000004_GM" : 1.7289999999999999E+01 ,
        "BODY2000006_GM" : 9.3000000000000005E-01 ,
        "BODY2000007_GM" : 8.5999999999999999E-01 ,
        "BODY2000010_GM" : 5.7800000000000002E+00 ,
        "BODY2000015_GM" : 2.1000000000000001E+00 ,
        "BODY2000016_GM" : 1.8100000000000001E+00 ,
        "BODY2000029_GM" : 8.5999999999999999E-01 ,
        "BODY2000052_GM" : 1.5900000000000001E+00 ,
        "BODY2000065_GM" : 9.1000000000000003E-01 ,
        "BODY2000087_GM" : 9.8999999999999999E-01 ,
        "BODY2000088_GM" : 1.0200000000000000E+00 ,
        "BODY2000433_GM" :  4.463E-4 ,
        "BODY2000511_GM" : 2.2599999999999998E+00 ,
        "BODY2000704_GM" : 2.1899999999999999E+00 
    }
    return de431


