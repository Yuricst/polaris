#!/usr/bin/env python3
"""
Functions associated with Keplerian dynamics
"""


from ._rootSolve import __newtonRaphson
from ._rotationMatrices import *
from ._orbitalElements import *
from ._statevec2elements import sv2elts, print_elts
from ._elements2statevec import elts2sv
from ._miscExpressions import get_synodic_period, get_energy, get_c3, get_hohmann_cost


from ._kepler_propagator import kepler_propagate
from ._keplerder import keplerder, hypertrig_s, hypertrig_c
