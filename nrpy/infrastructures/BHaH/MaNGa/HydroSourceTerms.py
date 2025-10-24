"""
Construct expressions for Hydro Source Terms.

Author: Thiago Assumpcao
        assumpcaothiago **at** gmail **dot* com
"""

# Step 1.a: import all needed modules from NRPy+:

from inspect import currentframe as cfr
from pathlib import Path
from types import FrameType as FT
from typing import Dict, Tuple, Union, cast

import sympy as sp  # SymPy: The Python computer algebra package upon which NRPy+ depends

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.grid as gri
import nrpy.indexedexp as ixp  # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import nrpy.reference_metric as refmetric  # NRPy+: Reference metric support
import nrpy.helpers.parallel_codegen as pcg
import nrpy.infrastructures.BHaH.simple_loop as lp

import nrpy.helpers.jacobians as jac
from nrpy.equations.grhd.GRHD_equations import GRHD_Equations
from nrpy.equations.general_relativity.BSSN_quantities import BSSNQuantities


# Generate function to compute source term of momentum equation
def register_CFunction_compute_source_term_rescaled_sD(
    CoordSystem: str,
    enable_rfm_precompute: bool,
    enable_simd: bool,
    OMP_collapse: int,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register the the rescaled source term for momentum equation

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.
    :param enable_simd: Whether to enable SIMD.
    :param OMP_collapse: Level of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    if enable_simd:
        includes += [str(Path("intrinsics") / "simd_intrinsics.h")]
    desc = r"""Compute resclaed source term for momentum equation."""
    cfunc_type = "void"
    name = "compute_source_term_rescaled_sD"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const REAL *restrict auxevol_gfs, REAL *restrict in_gfs"
    if enable_rfm_precompute:
        params = params.replace(
            "REAL *restrict xx[3]", "const rfm_struct *restrict rfmstruct"
        )

    # Step 3: Register gridfunctions that are needed as input
    if "MaNGa_rescaled_sD0" not in gri.glb_gridfcs_dict:
        _MaNGa_rescaled_sD = gri.register_gridfunctions_for_single_rank1("MaNGa_rescaled_sD", group="EVOL")

    #  Register gridfunctions that are needed as input
    if "T4UU00" not in gri.glb_gridfcs_dict:
        _T4UU = gri.register_gridfunctions_for_single_rank2("T4UU", group="AUXEVOL", dimension=4, symmetry="sym01")

    # Compute GRHD variables
    grhd_eqs = GRHD_Equations(
        CoordSystem=CoordSystem, enable_rfm_precompute=enable_rfm_precompute
    )
    grhd_eqs.construct_all_equations()

    # Step 3: Compute rescaled source term of momentum equation

    rfm = refmetric.reference_metric[
        CoordSystem + "_rfm_precompute" if enable_rfm_precompute else CoordSystem
    ]

    rescaled_S_tilde_source_termD = ixp.zerorank1()
    for i in range(3):
        rescaled_S_tilde_source_termD[i] = grhd_eqs.S_tilde_source_termD[i] / rfm.ReD[i]

    # # Transform source term to Cartesian coordinates
    # sCartD = jac.basis_transform_vectorD_from_rfmbasis_to_Cartesian(
    #     CoordSystem=CoordSystem, src_vectorD=grhd_eqs.S_tilde_source_termD
    # )

    body = lp.simple_loop(
        loop_body=ccg.c_codegen(
            rescaled_S_tilde_source_termD,
            [
                gri.BHaHGridFunction.access_gf(
                    f"MaNGa_rescaled_sD{i}", gf_array_name="in_gfs"
                )
                for i in range(3)
            ],
            enable_fd_codegen=True,
            enable_simd=enable_simd,
        ),
        loop_region="interior",
        enable_intrinsics=enable_simd,
        CoordSystem=CoordSystem,
        enable_rfm_precompute=enable_rfm_precompute,
        read_xxs=not enable_rfm_precompute,
        OMP_collapse=OMP_collapse,
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
        enable_simd=enable_simd,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


# Declare function to compute derivatives of the lapse


def register_CFunction_compute_rescaled_alpha_dD(
    CoordSystem: str,
    enable_rfm_precompute: bool,
    enable_simd: bool,
    OMP_collapse: int,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function that computes rescaled derivatives of the lapse function.

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.
    :param enable_simd: Whether to enable SIMD.
    :param OMP_collapse: Level of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    if enable_simd:
        includes += [str(Path("intrinsics") / "simd_intrinsics.h")]
    desc = r"""Compute rescaled derivatives of the lapse function."""
    cfunc_type = "void"
    name = "compute_rescaled_alpha_dD"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], REAL *restrict in_gfs"
    if enable_rfm_precompute:
        params = params.replace(
            "REAL *restrict xx[3]", "const rfm_struct *restrict rfmstruct"
        )

    # Step 1: Register gridfunctions that are needed as input
    if "MaNGa_rescaled_alphaD" not in gri.glb_gridfcs_dict:
        _MaNGa_rescaled_alphaD = gri.register_gridfunctions_for_single_rank1(
            "MaNGa_rescaled_alphaD",
            dimension=3,
            group="EVOL",
        )

    # Step 2: Set up BSSN quantities
    _bssn = BSSNQuantities(
        CoordSystem=CoordSystem, enable_rfm_precompute=enable_rfm_precompute
    )

    # Step 3: Compute derivatives of lapse function
    alpha_dD = ixp.declarerank1("alpha_dD")

    rfm = refmetric.reference_metric[
        CoordSystem + "_rfm_precompute" if enable_rfm_precompute else CoordSystem
    ]

    rescaled_alpha_dD = ixp.zerorank1()
    for i in range(3):
        rescaled_alpha_dD[i] = alpha_dD[i] / rfm.ReD[i]

    # fmt: off
    body = lp.simple_loop(
        loop_body=ccg.c_codegen(
            rescaled_alpha_dD,
            [gri.BHaHGridFunction.access_gf(f"MaNGa_rescaled_alphaD{i}", gf_array_name="in_gfs") for i in range(3)],
            enable_fd_codegen=True,
            enable_simd=enable_simd,
        ),
        loop_region="interior",
        enable_intrinsics=enable_simd,
        CoordSystem=CoordSystem,
        enable_rfm_precompute=enable_rfm_precompute,
        read_xxs=not enable_rfm_precompute,
        OMP_collapse=OMP_collapse,
    )
    # fmt: on

    cfc.register_CFunction(
        include_CodeParameters_h=True,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        body=body,
        enable_simd=enable_simd,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


#######################################################################################
def register_CFunction_compute_vectorCartD_from_rescaled_vectorD(
    CoordSystem: str,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function that unrescales vector in rfm basis and basis transforms to Cartesian basis

    :param CoordSystem: The coordinate system.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    desc = r"""Unrescales vector in rfm basis and basis transforms to Cartesian basis."""
    cfunc_type = "void"
    name = "compute_vectorCartD_from_rescaled_vectorD"
    params = r"""const commondata_struct *restrict commondata, const params_struct *restrict params,
        const REAL xx0, const REAL xx1, const REAL xx2,
        const REAL rescaled_vectorD0, const REAL rescaled_vectorD1, const REAL rescaled_vectorD2,
        REAL *vectorCartD0, REAL *vectorCartD1, REAL *vectorCartD2"""

    # Step 1: Set up indexed expression for rescaled derivatives of alpha
    rescaled_vectorD = ixp.declarerank1("rescaled_vectorD")

    # Step 2: Set up reference metric
    rfm = refmetric.reference_metric[CoordSystem]

    # Step 3: Compute unrescaled derivatives
    vectorD = ixp.zerorank1(dimension=3)
    for i in range(3):
        vectorD[i] = rescaled_vectorD[i] * rfm.ReD[i]

    # Step 4: Compute derivatives of alpha in Cartesian basis from unrescaled derivatives
    vectorCartD = jac.basis_transform_vectorD_from_rfmbasis_to_Cartesian(
        CoordSystem=CoordSystem, src_vectorD=vectorD
    )

    body = ccg.c_codegen(
        vectorCartD,
        [f"*vectorCartD{i}" for i in range(3)],
        enable_fd_codegen=False,
        enable_simd=False,
        include_braces=False,
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

#######################################################################################
def register_CFunction_interpolate_MaNGa_variables_Cart_basis(enable_simd=False) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function that interpolates MaNGa variables in Cartesian basis at an arbitrary point (Cartx, Carty, Cartz).

    :param enable_simd: Whether to enable SIMD.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = r"""Interpolate MaNGa variables in Cartesian basis at an arbitrary point (Cartx, Carty, Cartz)."""
    cfunc_type = "void"
    name = "interpolate_MaNGa_variables_Cart_basis"
    params = r"""const commondata_struct *restrict commondata, griddata_struct *restrict griddata,
        const REAL Cartx, const REAL Carty, const REAL Cartz,
        REAL *alphaCart, REAL *alphaCartD0, REAL *alphaCartD1, REAL *alphaCartD2,
        REAL *sCartD0, REAL *sCartD1, REAL *sCartD2"""

    body =r"""
  // Unpack griddata
  const int grid = 0; // single grid
  params_struct *restrict params = &griddata[grid].params;

#include "set_CodeParameters-simd.h"

  // Set grid coordinates
  REAL *restrict xx[3] = {griddata[grid].xx[0], griddata[grid].xx[1], griddata[grid].xx[2]};

  // Set evolution grid functions
  REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;

  // Map Cartesian -> reference-metric
  REAL xCart[3] = {Cartx, Carty, Cartz};
  REAL xx012[3];
  int i012[3];
  Cart_to_xx_and_nearest_i0i1i2(commondata, params, xCart, xx012, i012);

  // interpolation setup
  const int num_interp_gfs = 7;
  static const int list_of_interp_gfs[7] = {ALPHAGF, MANGA_RESCALED_ALPHAD0GF, MANGA_RESCALED_ALPHAD1GF, MANGA_RESCALED_ALPHAD2GF,
                                            MANGA_RESCALED_SD0GF, MANGA_RESCALED_SD1GF, MANGA_RESCALED_SD2GF};
  const int num_interp_pts = 1;
  const int N_interp_GHOSTS = 3; // interpolation order = 2 * N_interp_GHOSTS + 1
  const REAL src_dxx0 = params->dxx0;
  const REAL src_dxx1 = params->dxx1;
  const REAL src_dxx2 = params->dxx2;

  // Single destination point array
  REAL dst_x0x1x2[1][3] = {{xx012[0], xx012[1], xx012[2]}};

  // Allocate interpolation buffers
  REAL interp_output[num_interp_gfs * num_interp_pts];
  const REAL *restrict src_gf_ptrs[num_interp_gfs];
  REAL *restrict dst_data[num_interp_gfs];

  // Compute total pts per GF
  int Nxx_plus_2NGHOSTS_TOTAL = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  // Build pointers into source and destination buffers
  for (int gf = 0; gf < num_interp_gfs; gf++) {
    int which = list_of_interp_gfs[gf];
    src_gf_ptrs[gf] = &y_n_gfs[which * Nxx_plus_2NGHOSTS_TOTAL];
    dst_data[gf] = &interp_output[gf * num_interp_pts];
  }

  // Call the new interpolator
  int err = interpolation_3d_general__uniform_src_grid(N_interp_GHOSTS, src_dxx0, src_dxx1, src_dxx2, Nxx_plus_2NGHOSTS0, Nxx_plus_2NGHOSTS1,
                                                       Nxx_plus_2NGHOSTS2, num_interp_gfs, xx, src_gf_ptrs, num_interp_pts, dst_x0x1x2, dst_data);
  if (err != 0) {
    fprintf(stderr, "computer_alphaCart_and_derivatives: interpolation error %d\n", err);
    exit(1);
  }

  // Unpack interpolated values at a single point
  *alphaCart = dst_data[0][0];
  // Rescaled derivatives of lapse
  const REAL rescaled_alphaD0 = dst_data[1][0];
  const REAL rescaled_alphaD1 = dst_data[2][0];
  const REAL rescaled_alphaD2 = dst_data[3][0];
  // Rescaled source terms to momentum equation
  const REAL rescaled_sD0 = dst_data[4][0];
  const REAL rescaled_sD1 = dst_data[5][0];
  const REAL rescaled_sD2 = dst_data[6][0];

  // Unrescaled deriavtives of alpha and transform vector to Cartesian basis
  compute_vectorCartD_from_rescaled_vectorD(commondata, params, xx012[0], xx012[1], xx012[2], rescaled_alphaD0, rescaled_alphaD1, rescaled_alphaD2,
                                            alphaCartD0, alphaCartD1, alphaCartD2);

  // Unrescaled source term and transform vector to Cartesian basis
  compute_vectorCartD_from_rescaled_vectorD(commondata, params, xx012[0], xx012[1], xx012[2], rescaled_sD0, rescaled_sD1, rescaled_sD2,
                                            sCartD0, sCartD1, sCartD2);
"""
    if not enable_simd:
        body = body.replace("set_CodeParameters-simd.h", "set_CodeParameters.h")

    cfc.register_CFunction(
        include_CodeParameters_h=False,  # already manually included
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        body=body,
        enable_simd=enable_simd,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


if __name__ == "__main__":

    register_CFunction_compute_source_term_rescaled_sD(
        CoordSystem="SinhCylindrical",
        enable_rfm_precompute=False,
        enable_simd=False,
        OMP_collapse=1,
    )
    # Print function
    print(cfc.CFunction_dict["compute_source_term_rescaled_sD"].full_function)

    # register_CFunction_compute_rescaled_alpha_dD(
    #     CoordSystem="SinhCylindrical",
    #     enable_rfm_precompute=False,
    #     enable_simd=False,
    #     OMP_collapse=1,
    # )
    # # Print function
    # print(cfc.CFunction_dict["compute_rescaled_alpha_dD"].full_function)

    # register_CFunction_compute_vectorCartD_from_rescaled_vectorD(CoordSystem="SinhCylindrical")
    # # Print function
    # print(cfc.CFunction_dict["compute_vectorCartD_from_rescaled_vectorD"].full_function)
