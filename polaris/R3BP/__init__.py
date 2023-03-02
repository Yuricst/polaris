#!/usr/bin/env python3
"""
Functions associated with Restricted Three Body Problem
"""

# functions to get characteristic values of r3bp system
from ._define_r3bp_param import (
    CR3BP,
    BCR4BP,
    get_cr3bp_mu,
    get_bcr4bp_mus,
    get_cr3bp_param,
    get_bcr4bp_param,
)


# functions to compute characterstic values
from ._lagrangePoints import lagrangePoints
from ._jacobiConstant import jacobiConstant
from ._stabilityIndex import stabilityIndex

# manifold functions
from ._manifold import get_manifold, _get_eigvecs_yu_ys

# differential correction functions
from ._ssdc_periodic_xzplane import ssdc_periodic_xzplane


# halo in 3rd order approximation
from ._approxHalo3rdOrder import get_halo_approx

# jacobi contours
from ._jacobiContour import get_jacobi_contour

# circularization cost
from ._circularize_cost import get_loi_cost_inertial, get_loicost


# function for triangular libration points
from ._approxTriangularLibPoints import get_linearmotion_ic
