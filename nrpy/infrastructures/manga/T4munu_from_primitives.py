# nrpy/infrastructures/manga/T4munu_from_primitives.py
"""
Construct C function to compute T4UU[mu][nu] in curvilinear coordinates in a point-wise fashion

Author: Thiago Assumpcao
        assumpcaothiago **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
import nrpy.grid as gri
from nrpy.equations.grhd.GRHD_equations import GRHD_Equations


def register_CFunction_compute_T4UU(
    CoordSystem: str,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Compute T4UU[mu][nu] in curvilinear coordinates in a point-wise fashion

    :param CoordSystem: The coordinate system.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    desc = r"""Compute T4UU in curvilinear coordinates in a point-wise fashion."""
    cfunc_type = "void"
    name = "compute_T4UU"
    params = r"""const commondata_struct *restrict commondata, const params_struct *restrict params,
    const int i0, const int i1, const int i2,
    REAL *restrict xx[3], const REAL rhob, const REAL P, const REAL h, const REAL u4Ut,
    const REAL rescaledvU0, const REAL rescaledvU1, const REAL rescaledvU2, const REAL *restrict in_gfs, REAL *restrict auxevol_gfs"""

    # Step 1: Register gridfunctions that are needed as input
    if "T4UU00" not in gri.glb_gridfcs_dict:
        _T4UU = gri.register_gridfunctions_for_single_rank2(
            "T4UU", group="AUXEVOL", dimension=4, symmetry="sym01"
        )

    # Step 2: Compute GRHD variables
    grhd_eqs = GRHD_Equations(CoordSystem=CoordSystem)
    grhd_eqs.construct_all_equations()

    # Step 3: Set up expressions and symbols
    sympy_exprs = []
    varnames = []
    for mu in range(4):
        for nu in range(mu, 4):
            sympy_exprs.append(grhd_eqs.T4UU[mu][nu])
            varnames.append(
                gri.BHaHGridFunction.access_gf(
                    f"T4UU{mu}{nu}", gf_array_name="auxevol_gfs"
                )
            )

    body = r"""  // Set up coordinates at (i0, i1, i2)
    MAYBE_UNUSED REAL xx0 = xx[0][i0];
    MAYBE_UNUSED REAL xx1 = xx[1][i1];
    MAYBE_UNUSED REAL xx2 = xx[2][i2];
    """

    body += ccg.c_codegen(
        sympy_exprs,
        varnames,
        verbose=False,
        include_braces=False,
        enable_fd_codegen=True,
        enable_simd=False,
    )

    cfc.register_CFunction(
        include_CodeParameters_h=True,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        body=body,
        enable_simd=False,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())
