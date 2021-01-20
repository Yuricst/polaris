"""
Propagation using multiprocessing
"""


from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from tqdm.notebook import tqdm
import time

import sys
sys.path.append('../')   # path to polaris module

import polaris.SolarSystemConstants as sscs
import polaris.Keplerian as kepl
import polaris.Propagator as prop
import polaris.R3BP as r3bp

class Propagation:
    def __init__(self, mu, state, tf):
        self.mu = mu
        self.state = state
        self.tf =tf

    def run(self):
        propout = prop.propagate_cr3bp(self.mu, self.state, self.tf)
        xfinal = propout["statef"][0]
        #print(f"Final state-x: {xfinal}")


# -------------------------------------------------------------------------------------------------- #
# main bit
if __name__ == "__main__":
    # ---------------------------------------------------- #
    # parameters
    param_earth_moon = r3bp.get_cr3bp_param('399','301')   # NAIF ID's '399': Earth, '301': Moon
    lp = r3bp.lagrangePoints(param_earth_moon.mu)

    # ---------------------------------------------------- #
    # generate initial guess
    ics, ps = [], []
    azs = np.linspace(2000, 4000, 20)
    for az in tqdm(azs):
        haloinit = r3bp.get_halo_approx(mu=param_earth_moon.mu, lp=2, lstar=param_earth_moon.lstar, az_km=4000, family=1, phase=0.0)

        p_conv, state_conv, flag_conv = r3bp.ssdc_periodic_xzplane(param_earth_moon.mu, haloinit["state_guess"], 
                                                                   haloinit["period_guess"], fix="z", message=False)

        ics.append(state_conv)
    
    ps = np.linspace(0.98, 10, 100)*p_conv

    # ---------------------------------------------------- #
    print(" =================== Single processing =================== ")
    start_s = time.time()
    for idx, ic in enumerate(ics):
        for p in ps:
            propout = prop.propagate_cr3bp(param_earth_moon.mu, ic, p)
            xfinal = propout["statef"][0]
        #print(f"Final state-x: {xfinal}")
    end_s = time.time()
    delta_s = end_s - start_s
    print('処理時間:{}s'.format(round(delta_s, 4))) 

    # ---------------------------------------------------- #
    print(" =================== Multiple processing =================== ")
    start_mp = time.time()
    process_pool = []
    for idx, ic in enumerate(ics):
        for p in ps:
            process_pool.append( Propagation(param_earth_moon.mu, ic, p).run )

    # run multiple processing
    with ProcessPoolExecutor(max_workers=4) as executor: # max_workerは同時に動かすプロセスの最大数 Noneを指定するとコア数 * 4の値になる
        for process in process_pool:
            results = executor.submit(process)

    end_mp = time.time()
    delta_mp = end_mp - start_mp
    print('処理時間:{}s'.format(round(delta_mp, 4))) 

    print("Done!")

