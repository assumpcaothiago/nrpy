"""
Construct symbolic expressions for the right-hand-side of the hyperbolic relaxation equation.
Curvilinear coordinates are supported using a reference-metric formalism.

Authors: Thiago Assumpção; assumpcaothiago **at** gmail **dot* com
         Zachariah B. Etienne; zachetie **at** gmail **dot* com

License: BSD 2-Clause
"""

# Step P1: Import needed modules:
from collections import OrderedDict
from typing import Dict

import sympy as sp  # For symbolic computations

import nrpy.grid as gri  # NRPy+: Functionality for handling numerical grids
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.reference_metric as refmetric  # NRPy+: Reference metric support
from nrpy.equations.nrpyelliptic.CommonParams import eta_damping


# Specify RHSs as class variables,
# to enable access outside this
# function (e.g., for C code output)
class HyperbolicRelaxationCurvilinearRHSs:
    """Class sets up and stores sympy expressions for wave equation RHSs in curvilinear coordinates."""

    def __init__(self, CoordSystem: str, enable_rfm_precompute: bool) -> None:
        """
        Compute the right-hand sides (RHSs) of the hyperbolic relaxation equation in curvilinear coordinates.

        :param CoordSystem: The coordinate system being used.
        :param enable_rfm_precompute: Whether to enable reference metric precomputation.

        .. note::
            Class variables psi_rhs, xi_rhs, and residual will be set in this function.
        """
        # Step 1: Set up the reference metric and
        #         quantities derived from the
        #         reference metric.
        rfm = refmetric.reference_metric[
            CoordSystem + "_rfm_precompute" if enable_rfm_precompute else CoordSystem
        ]

        # Step 2: Compute the contracted Christoffel symbols:
        contractedGammahatU = ixp.zerorank1()
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    contractedGammahatU[k] += (
                        rfm.ghatUU[i][j] * rfm.GammahatUDD[k][i][j]
                    )

        # Step 3: Register gridfunctions that are needed as input
        #         to the scalar wave RHS expressions.
        if "psi" not in gri.glb_gridfcs_dict:
            psi, xi, alphaconf, tau = gri.register_gridfunctions(
                ["psi", "xi", "alphaconf", "tau"],
                group="EVOL",
                f_infinity=[0.0, 0.0, 0.0, 0.0],
                wavespeed=[1.0, 1.0, 1.0, 1.0],
            )
        else:
            psi, xi, alphaconf, tau = sp.symbols("psi, xi, alphaconf, tau", real=True)

        # Step 4a: Declare the rank-1 indexed expressions \partial_{i},
        #          Derivative variables like these must have an underscore
        #          in them, so the finite difference module can parse the
        #          variable name properly.
        psi_dD = ixp.declarerank1("psi_dD")
        alphaconf_dD = ixp.declarerank1("alphaconf_dD")

        # Step 4b: Declare the rank-2 indexed expression \partial_{ij},
        #          which is symmetric about interchange of indices i and j
        #          Derivative variables like these must have an underscore
        #          in them, so the finite difference module can parse the
        #          variable name properly.
        psi_dDD = ixp.declarerank2("psi_dDD", symmetry="sym01")
        alphaconf_dDD = ixp.declarerank2("alphaconf_dDD", symmetry="sym01")

        # Step 5.a: Declare spatially-dependent variable_wavespeed as grid function
        if "variable_wavespeed" not in gri.glb_gridfcs_dict:
            variable_wavespeed = gri.register_gridfunctions(
                ["variable_wavespeed"], group="AUXEVOL", gf_array_name="auxevol_gfs"
            )[0]
        else:
            variable_wavespeed = sp.Symbol("variable_wavespeed", real=True)

        # Step 5.b: Register AUXEVOL gridfunctions that are part of RHSs
        if "rho" not in gri.glb_gridfcs_dict:
            rho, P = gri.register_gridfunctions(
                ["rho", "P"],
                group="AUXEVOL",
                gf_array_name="auxevol_gfs",
            )
        else:
            rho, P = sp.symbols("rho P", real=True)

        # Step 5.c: Register residuals as AUX gridfunction
        if "residual_psi" not in gri.glb_gridfcs_dict:
            _residual_psi, _residual_alphaconf = gri.register_gridfunctions(
                ["residual_psi", "residual_alphaconf"],
                group="AUX",
                gf_array_name="aux_gfs",
            )

        # Step 6: Define right-hand sides for the evolution.

        # Step 6a: The RHSs of fundamental variables
        self.psi_rhs = xi - eta_damping * psi
        self.alphaconf_rhs = tau - eta_damping * alphaconf

        # Step 6b: The RHSs of the time reduction variables

        # The Laplacian of both variables u = {psi, alphaconf} is given by:
        #          \hat{g}^{ij} \partial_i \partial_j u - \hat{\Gamma}^i \partial_i u.
        #          ^^^^^^^^^^^^ PART 1 ^^^^^^^^^^^^^^^^ ^^^^^^^^^^ PART 2 ^^^^^^^^^^^
        self.xi_rhs = sp.sympify(0)
        self.tau_rhs = sp.sympify(0)
        for i in range(3):
            # PART 2:
            self.xi_rhs -= contractedGammahatU[i] * psi_dD[i]
            self.tau_rhs -= contractedGammahatU[i] * alphaconf_dD[i]
            for j in range(3):
                # PART 1:
                self.xi_rhs += rfm.ghatUU[i][j] * psi_dDD[i][j]
                self.tau_rhs += rfm.ghatUU[i][j] * alphaconf_dDD[i][j]

        # Step 7.a : Add source terms
        self.xi_rhs += 2 * sp.pi * ((1 + psi) ** 5) * rho
        self.tau_rhs -= 2 * sp.pi * (1 + alphaconf) * ((1 + psi) ** 4) * (rho + 6 * P)

        # Step 8: Set residual before multiplying xi_rhs and tau_rhs by variable_wavespeed**2
        self.residual_psi = self.xi_rhs
        self.residual_alphaconf = self.tau_rhs

        # Step 9: Multiply xi_rhs and tau_rhs by variable_wavespeed**2 according to the hyperbolization precription
        self.xi_rhs *= variable_wavespeed * variable_wavespeed
        self.tau_rhs *= variable_wavespeed * variable_wavespeed

        # Step 10: Create dictionary that maps variable names to symbolic expressions
        self.NRPyElliptic_RHSs_varname_to_expr_dict: Dict[str, sp.Expr] = OrderedDict()
        self.NRPyElliptic_RHSs_varname_to_expr_dict["psi_rhs"] = self.psi_rhs
        self.NRPyElliptic_RHSs_varname_to_expr_dict["alphaconf_rhs"] = (
            self.alphaconf_rhs
        )
        self.NRPyElliptic_RHSs_varname_to_expr_dict["xi_rhs"] = self.xi_rhs
        self.NRPyElliptic_RHSs_varname_to_expr_dict["tau_rhs"] = self.tau_rhs
        # Sort the lists alphabetically by varname:
        self.NRPyElliptic_RHSs_varname_to_expr_dict = OrderedDict(
            sorted(self.NRPyElliptic_RHSs_varname_to_expr_dict.items())
        )


if __name__ == "__main__":
    import doctest
    import os
    import sys

    import nrpy.validate_expressions.validate_expressions as ve

    results = doctest.testmod()
    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")

    for Coord in [
        "Cartesian",
        "SinhCartesian",
        "Spherical",
        "SinhSpherical",
        "Cylindrical",
        "SinhCylindrical",
        "SymTP",
        "SinhSymTP",
    ]:
        RHS = HyperbolicRelaxationCurvilinearRHSs(
            CoordSystem=Coord, enable_rfm_precompute=False
        )
        results_dict = ve.process_dictionary_of_expressions(
            RHS.__dict__, fixed_mpfs_for_free_symbols=True
        )
        ve.compare_or_generate_trusted_results(
            os.path.abspath(__file__),
            os.getcwd(),
            # File basename. If this is set to "trusted_module_test1", then
            #   trusted results_dict will be stored in tests/trusted_module_test1.py
            f"{os.path.splitext(os.path.basename(__file__))[0]}_{Coord}",
            results_dict,
        )
