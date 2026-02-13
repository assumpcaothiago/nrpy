# nrpy/infrastructures/manga/u4Ut.py
"""
Construct C function to compute the time-component of the four-velocity, in a point-wise fashion

Author: Terrence Pierre Jacques
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
from nrpy.equations.general_relativity.BSSN_quantities import BSSN_quantities
from nrpy.equations.general_relativity.BSSN_to_ADM import BSSN_to_ADM
import nrpy.equations.grhd.Min_Max_and_Piecewise_Expressions as noif


def register_CFunction_compute_u4Ut(
    CoordSystem: str,
    enable_rfm_precompute: bool = False,
    enable_GoldenKernels: bool = False,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to compute the time-component of the four-velocity, in a point-wise fashion

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.
    :param enable_GoldenKernels: Boolean to enable Golden Kernels.

    :return: None if in registration phase, else the updated NRPy environment.

    ::note: have to turn off simd, since this is a point wise operation
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = r"""
compute time component of four velocity, via

// Derivation of first equation:
// \gamma_{ij} (v^i + \beta^i)(v^j + \beta^j)/(\alpha)^2
//   = \gamma_{ij} 1/(u^0)^2 ( \gamma^{ik} u_k \gamma^{jl} u_l /(\alpha)^2 <- Using Eq. 53 of arXiv:astro-ph/0503420
//   = 1/(u^0 \alpha)^2 u_j u_l \gamma^{jl}  <- Since \gamma_{ij} \gamma^{ik} = \delta^k_j
//   = 1/(u^0 \alpha)^2 ( (u^0 \alpha)^2 - 1 ) <- Using Eq. 56 of arXiv:astro-ph/0503420
//   = 1 - 1/(u^0 \alpha)^2 <= 1
"""
    cfunc_type = "void"
    name = "compute_u4Ut"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL max_Lorentz_factor, "
    params += "const int i0, const int i1, const int i2, REAL *restrict in_gfs, "
    params += "REAL *restrict rescaledvU0, REAL *restrict rescaledvU1, REAL *restrict rescaledvU2, REAL *restrict u4Ut"

    body = r"""
    // 1 - W_max^{-2}
    const REAL inv_sq_max_Lorentz_factor = 1.0/SQR(max_Lorentz_factor);
    const REAL one_minus_one_over_W_max_squared = 1.0 - inv_sq_max_Lorentz_factor;

    const REAL rescaledvU_old0 = *rescaledvU0;
    const REAL rescaledvU_old1 = *rescaledvU1;
    const REAL rescaledvU_old2 = *rescaledvU2;
"""

    # ADM in terms of BSSN
    AitoB = BSSN_to_ADM(
        CoordSystem=CoordSystem, enable_rfm_precompute=enable_rfm_precompute
    )

    # Import all basic (unrescaled) BSSN scalars & tensors
    Bq = BSSN_quantities[
        CoordSystem + ("_rfm_precompute" if enable_rfm_precompute else "")
    ]

    rfm = refmetric.reference_metric[
        CoordSystem + "_rfm_precompute" if enable_rfm_precompute else CoordSystem
    ]

    rescaledvU = ixp.declarerank1("rescaledvU_old")

    VU = ixp.zerorank1()
    utildeU = ixp.zerorank1()
    for i in range(3):
        VU[i] = rescaledvU[i] * rfm.ReU[i]
        # utU[3] = {prims->vU[0] + ADM_metric->betaU[0], ...
        utildeU[i] = VU[i] + Bq.betaU[i]

    one_minus_one_over_alpha_u0_squared = sp.sympify(0.0)
    for i in range(3):
        for j in range(3):
            one_minus_one_over_alpha_u0_squared += (
                AitoB.gammaDD[i][j] * (VU[i] + Bq.betaU[i]) * (VU[j] + Bq.betaU[j])
            )

    one_minus_one_over_alpha_u0_squared /= Bq.alpha**2

    one_minus_one_over_alpha_u0_squared = sp.simplify(
        one_minus_one_over_alpha_u0_squared
    )

    one_minus_one_over_W_max_squared = sp.symbols("one_minus_one_over_W_max_squared")

    # prevent divide by zero in cse
    TINYDOUBLE = sp.symbols("TINYDOUBLE")
    correction_fac = sp.sqrt(
        one_minus_one_over_W_max_squared
        / (one_minus_one_over_alpha_u0_squared + TINYDOUBLE)
    )

    # coord_geq_bound(x,xstar):
    # Returns 1.0 if x >= xstar, 0.0 otherwise.
    check_too_fast = noif.coord_geq_bound(
        one_minus_one_over_alpha_u0_squared, one_minus_one_over_W_max_squared
    )
    # coord_less_bound(x,xstar):
    # Returns 1.0 if x < xstar, 0.0 otherwise.
    check_normal = noif.coord_less_bound(
        one_minus_one_over_alpha_u0_squared, one_minus_one_over_W_max_squared
    )

    # limit velocity, especially in reconstruction step
    # GRHayL code from GRHayL_Core/limit_v_and_compute_u0.c

    #   double one_minus_one_over_alpha_u0_squared = ghl_compute_vec2_from_vec3D(ADM_metric->gammaDD, utU)*ADM_metric->lapseinv2;
    #   /*** Limit velocity to GAMMA_SPEED_LIMIT ***/
    #   const double one_minus_one_over_W_max_squared = 1.0 - params->inv_sq_max_Lorentz_factor; // 1 - W_max^{-2}
    #   if(one_minus_one_over_alpha_u0_squared > one_minus_one_over_W_max_squared) {
    #     const double correction_fac = sqrt(one_minus_one_over_W_max_squared/one_minus_one_over_alpha_u0_squared);
    #     prims->vU[0] = utU[0]*correction_fac - ADM_metric->betaU[0];
    #     prims->vU[1] = utU[1]*correction_fac - ADM_metric->betaU[1];
    #     prims->vU[2] = utU[2]*correction_fac - ADM_metric->betaU[2];
    #     one_minus_one_over_alpha_u0_squared = one_minus_one_over_W_max_squared;
    #     speed_limited = 1;

    rescaledvU_new = ixp.zerorank1()
    for i in range(3):
        rescaledvU_new[i] = (
            check_too_fast
            * (
                sp.together(utildeU[i] / rfm.ReU[i]) * correction_fac
                - sp.together(Bq.betaU[i] / rfm.ReU[i])
            )
            + check_normal * rescaledvU[i]
        )

    one_minus_one_over_alpha_u0_squared = noif.min_noif(
        one_minus_one_over_W_max_squared, one_minus_one_over_alpha_u0_squared
    )

    alpha_u0 = 1.0 / sp.sqrt(1.0 - one_minus_one_over_alpha_u0_squared)

    u4Ut = alpha_u0 / Bq.alpha

    sympy_exprs = [u4Ut, rescaledvU_new[0], rescaledvU_new[1], rescaledvU_new[2]]
    varnames = ["*u4Ut", "*rescaledvU0", "*rescaledvU1", "*rescaledvU2"]

    body += ccg.c_codegen(
        sympy_exprs,
        varnames,
        # u4Ut,
        # "*u4Ut",
        verbose=False,
        include_braces=False,
        enable_fd_codegen=True,
        enable_simd=False,
        enable_GoldenKernels=enable_GoldenKernels,
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
