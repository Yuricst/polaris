"""
polaris
Python Library for Astrodynamics
Yuri Shimane, 2020
"""


# dependencies ----------------------------------------------------------- #
# Let users know if they're missing any of our hard dependencies
hard_dependencies = ("numpy", "matplotlib", "numba", "pandas", "scipy")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(f"{dependency}: {e}")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

# imports ---------------------------------------------------------------- #
from . import Propagator
from . import R3BP
from . import Keplerian
from . import Coordinates
from . import LinearOrbit
