"""
Construct C function to compute vU on rescaled basis from vCartU

Author: Thiago Assumpcao
        assumpcaothiago **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
import sympy as sp
import nrpy.indexedexp as ixp
import nrpy.reference_metric as refmetric
import nrpy.helpers.jacobians as jac


def register_CFunction_compute_rescaledvU_from_vCartU(
    CoordSystem: str,
    enable_rfm_precompute: bool = False,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to compute rescaled 3-vector in curvilinar coordinates from 3-vector in Cartesian coordinates

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.

    :return: None if in registration phase, else the updated NRPy environment.

    ::note: have to turn off simd, since this is a point wise operation
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "Compute rescaled 3-vector in curvilinar coordinates from 3-vector in Cartesian coordinates."

    cfunc_type = "void"
    name = "compute_rescaledvU_from_vCartU"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL vCartU[3], "
    params += "const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict rescaledvU0, REAL *restrict rescaledvU1, REAL *restrict rescaledvU2"

    # Compute reference-metric quantities
    rfm = refmetric.reference_metric[
        CoordSystem + "_rfm_precompute" if enable_rfm_precompute else CoordSystem
    ]

    # Define symbol for 3-vector in Cartesian coordinates
    vCartU = ixp.zerorank1()
    for i in range(3):
        vCartU[i] = sp.Symbol(f"vCartU[{i}]")

    # Compute vector in curvilinear coordinates
    vU = jac.basis_transform_vectorU_from_Cartesian_to_rfmbasis(
        CoordSystem=CoordSystem, Cart_src_vectorU=vCartU
    )

    # Compute rescaled vector in curvilinear coordinates
    rescaledvU = ixp.zerorank1()
    for i in range(3):
        rescaledvU[i] = vU[i] / rfm.ReU[i]

    sympy_exprs = [rescaledvU[0], rescaledvU[1], rescaledvU[2]]
    varnames = ["*rescaledvU0", "*rescaledvU1", "*rescaledvU2"]

    body = ccg.c_codegen(
        sympy_exprs,
        varnames,
        verbose=False,
        include_braces=False,
        enable_fd_codegen=False,
        enable_simd=False,
    )

    cfc.register_CFunction(
        include_CodeParameters_h=True,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func=CoordSystem,
        name=name,
        params=params,
        body=body,
        enable_simd=False,
    )
    return pcg.NRPyEnv()
