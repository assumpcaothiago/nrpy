"""Reconstruct d/dx h_{theta theta} from stored spherical partials."""

import sympy as sp

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.reference_metric as refmetric


def register_CFunction_counterexample2_compute_dhdx() -> None:
    """Register the basis transform from spherical partials to d/dx."""
    rfm = refmetric.reference_metric["Spherical"]
    hdd11_d0, hdd11_d1, hdd11_d2 = sp.symbols(
        "hdd11_d0 hdd11_d1 hdd11_d2", real=True
    )
    dhdx_expr = sp.simplify(
        rfm.Jac_dUrfm_dDCartUD[0][0] * hdd11_d0
        + rfm.Jac_dUrfm_dDCartUD[1][0] * hdd11_d1
        + rfm.Jac_dUrfm_dDCartUD[2][0] * hdd11_d2
    )
    body = "  REAL dhdx;\n"
    body += ccg.c_codegen(
        [dhdx_expr],
        ["dhdx"],
        include_braces=False,
        verbose=False,
        enable_fd_codegen=False,
        enable_simd=False,
    )
    body += "  return dhdx;\n"
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Transform stored spherical coordinate-partial derivatives of h_{theta theta} into d/dx.",
        cfunc_type="REAL",
        name="counterexample2_compute_dhdx",
        params=(
            "const REAL xx0, const REAL xx1, const REAL xx2, "
            "const REAL hdd11_d0, const REAL hdd11_d1, const REAL hdd11_d2"
        ),
        include_CodeParameters_h=False,
        body=body,
    )
