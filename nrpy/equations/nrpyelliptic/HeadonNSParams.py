"""
Head-on NS parameters for NRPyElliptic.

Author: Thiago Assumpção
        assumpcaothiago **at** gmail **dot* com

License: BSD 2-Clause
"""

# Step P1: Import needed NRPy+ core modules:
import nrpy.params as par  # NRPy+: Parameter interface

# Parameters specific to Head-on NS problem

star_radius = par.register_CodeParameter(
    "REAL", __name__, "star_radius", 0.8100085557410306, commondata=True
)

rho_central = par.register_CodeParameter(
    "REAL", __name__, "rho_central", 0.1459996111924645, commondata=True
)

n_rho = par.register_CodeParameter(
    "REAL", __name__, "n_rho", 1.7339250800463264, commondata=True
)


sigma_rho = par.register_CodeParameter(
    "REAL", __name__, "sigma_rho", 0.5184929551053847, commondata=True
)

P_central = par.register_CodeParameter(
    "REAL", __name__, "P_central", 0.01671461121831567, commondata=True
)


n_P = par.register_CodeParameter(
    "REAL", __name__, "n_P", 1.9380083279682516, commondata=True
)


sigma_P = par.register_CodeParameter(
    "REAL", __name__, "sigma_P", 0.17181675341296368, commondata=True
)
