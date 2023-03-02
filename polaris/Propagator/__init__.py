#!/usr/bin/env python3
"""
Propagator and associated functions
Yuri Shimane
"""


# right-hand side functions
from ._rhs_functions import (
    rhs_twobody,
    rhs_twobody_with_STM,
    rhs_cr3bp,
    rhs_pcr3bp,
    rhs_cr3bp_with_STM,
    rhs_pcr3bp_with_STM,
    rhs_bcr4bp,
    rhs_bcr4bp_with_STM,
    rhs_ccr4bp,
    rhs_twobody_j2,
)


# integrators
from ._propagator_pcr3bp import (
    propagate_pcr3bp,
    propagate_pcr3bp_odeint,
    propagate_pcr3bp_solve_ivp,
)
from ._propagator_cr3bp import (
    propagate_cr3bp,
    propagate_cr3bp_odeint,
    propagate_cr3bp_solve_ivp,
)
from ._propagator_twobody import (
    propagate_twobody,
    propagate_twobody_odeint,
    propagate_twobody_solve_ivp,
)
from ._propagator_twobody_j2 import (
    propagate_twobody_j2,
    propagate_twobody_odeint_j2,
    propagate_twobody_solve_ivp_j2,
)
from ._propagator_bcr4bp import (
    propagate_bcr4bp,
    propagate_bcr4bp_odeint,
    propagate_bcr4bp_solve_ivp,
)
from ._propagator_ccr4bp import (
    propagate_ccr4bp,
    propagate_ccr4bp_odeint,
    propagate_ccr4bp_solve_ivp,
)
from ._interpolation import (
    prepare_interpolation,
    evaluate_interpolation,
    evaluate_time_grid
)

# single-shooting differential correction
from ._ssdc import ssdc
