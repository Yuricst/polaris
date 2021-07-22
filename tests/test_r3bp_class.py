"""
Test R3BP class constructions
"""


import sys
sys.path.append('../')   # path to polaris module

import polaris.SolarSystemConstants as sscs
import polaris.Keplerian as kepl
import polaris.Propagator as prop
import polaris.R3BP as r3bp


# create CR3BP system
EarthMoonSystem = r3bp.CR3BP(399, 301)
print(EarthMoonSystem)

EarthMoonSunSystem = r3bp.BCR4BP(399, 301)
print(EarthMoonSystem)
