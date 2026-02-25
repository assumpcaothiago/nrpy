# nrpy/infrastructures/manga/BHaH_get_metric_extrinsic_curvature.py
"""
Generate function to compute ADM variables in Cartesian basis at a single point.

Author: Thiago Assumpção
"""

from collections import OrderedDict
from typing import Dict, Union, cast
from inspect import currentframe as cfr
from types import FrameType as FT

import sympy as sp

import nrpy.c_function as cfc
import nrpy.c_codegen as ccg
import nrpy.helpers.parallel_codegen as pcg
import nrpy.helpers.jacobians as jac
from nrpy.equations.general_relativity.BSSN_to_ADM import BSSN_to_ADM


def _generate_body_to_compute_ADM_variables(CoordSystem, enable_rfm_precompute) -> str:
    """
    Generate function body that computed ADM variables from BSSN variables.

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.

    :return: String with function body.
    """

    # Step 1: Compute ADM variables in terms of BSSN variables, both in curvilinear coordinates
    BtoA = BSSN_to_ADM(
        CoordSystem=CoordSystem, enable_rfm_precompute=enable_rfm_precompute
    )

    # Step 2: Compute ADM variables in Cartesian coordinates
    betaCartU = jac.basis_transform_vectorU_from_rfmbasis_to_Cartesian(
        CoordSystem, BtoA.betaU
    )
    gammaCartDD = jac.basis_transform_tensorDD_from_rfmbasis_to_Cartesian(
        CoordSystem, BtoA.gammaDD
    )
    KCartDD = jac.basis_transform_tensorDD_from_rfmbasis_to_Cartesian(
        CoordSystem, BtoA.KDD
    )

    # Step 3.a: Create dictionary that maps variable names to symbolic expressions
    ADM_Cart_varname_to_expr_dict: Dict[str, sp.Expr] = OrderedDict()
    for i in range(3):
        ADM_Cart_varname_to_expr_dict[f"*betaCartU{i}"] = betaCartU[i]
        for j in range(3):
            ADM_Cart_varname_to_expr_dict[f"gammaCartDD[{i}][{j}]"] = gammaCartDD[i][j]
            ADM_Cart_varname_to_expr_dict[f"KCartDD[{i}][{j}]"] = KCartDD[i][j]
    # Step 3.b: Sort the lists alphabetically by varname:
    ADM_Cart_varname_to_expr_dict = OrderedDict(
        sorted(ADM_Cart_varname_to_expr_dict.items())
    )
    # Step 3.c: Add dictionary keys and values to define sympy expressions and variable names
    sympy_exprs = list(ADM_Cart_varname_to_expr_dict.values())
    varnames = list(ADM_Cart_varname_to_expr_dict.keys())

    # Step 4: Generate body of C expression
    body = ccg.c_codegen(
        sympy_exprs,
        varnames,
        verbose=False,
        include_braces=True,
        enable_fd_codegen=False,
        enable_simd=False,
    )

    return body


def register_CFunction_BHaH_get_metric_extrinsic_curvature(
    CoordSystem,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to compute ADM variables in Cartesian basis at a single point (Cartx, Carty, Cartz).

    :param CoordSystem: The coordinate system.

    :return: None if in registration phase, else the updated NRPy environment.

    ::note: have to turn off enable_rfm_precompute, since gridpoints are needed for interpolation
    ::note: have to turn off simd, since this is a point wise operation
    """

    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "Compute metric and extrinsic curvature in Cartesian basis at a single point (Cartx, Carty, Cartz)."

    cfunc_type = "void"
    name = "BHaH_get_metric_extrinsic_curvature"
    params = r"""BHaH_struct *bhahstruct, REAL Cartx, REAL Carty, REAL Cartz,
  REAL *alpha, REAL *betaCartU0, REAL *betaCartU1, REAL *betaCartU2,
  REAL gammaCartDD[3][3], REAL KCartDD[3][3]"""

    body = r"""
  // Step 1: Unpack data from bhahstruct

  // Step 1.a: commondata & griddata
  commondata_struct *commondata = bhahstruct->commondata;
  griddata_struct *restrict griddata = bhahstruct->griddata;
  const int grid = 0; // single grid
  params_struct *restrict params = &griddata[grid].params;

  // Step 1.b: Set grid coordinates
  REAL *restrict xx[3] = {griddata[grid].xx[0], griddata[grid].xx[1], griddata[grid].xx[2]};

  // Step 1.c: Set evolution grid functions
  REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;

#include "set_CodeParameters.h"

  // Step 2: Interpolate data onto point (Cartx, Carty, Cartz)

  // Step 2.a: Map Cartesian -> reference-metric
  REAL xCart[3] = {Cartx, Carty, Cartz};
  REAL xx012[3];
  int i012[3];
  Cart_to_xx_and_nearest_i0i1i2(params, xCart, xx012, i012);
  const REAL xx0 = xx012[0];
  const REAL xx1 = xx012[1];
  const REAL xx2 = xx012[2];

  // Step 2.b: Interpolation setup
  const int num_interp_gfs = 18;
  static const int list_of_interp_gfs[18] = {
      ALPHAGF, CFGF, TRKGF, VETU0GF, VETU1GF, VETU2GF,
      HDD00GF, HDD01GF, HDD02GF, HDD11GF, HDD12GF, HDD22GF,
      ADD00GF, ADD01GF, ADD02GF, ADD11GF, ADD12GF, ADD22GF};
  const int num_interp_pts = 1;
  const int N_interp_GHOSTS = NGHOSTS; // interpolation order = 2 * N_interp_GHOSTS + 1
  const REAL src_dxx0 = dxx0;
  const REAL src_dxx1 = dxx1;
  const REAL src_dxx2 = dxx2;
  REAL dst_x0x1x2[1][3] = {{xx0, xx1, xx2}}; // Single destination point array

  // Step 2.c: Allocate interpolation buffers
  REAL interp_output[num_interp_gfs * num_interp_pts];
  const REAL *restrict src_gf_ptrs[num_interp_gfs];
  REAL *restrict dst_data[num_interp_gfs];

  // Step 2.d: Compute total pts per GF
  int Nxx_plus_2NGHOSTS_TOTAL = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  // Step 2.e: Build pointers into source and destination buffers
  for (int gf = 0; gf < num_interp_gfs; gf++) {
    int which = list_of_interp_gfs[gf];
    src_gf_ptrs[gf] = &y_n_gfs[which * Nxx_plus_2NGHOSTS_TOTAL];
    dst_data[gf] = &interp_output[gf * num_interp_pts];
  }

  // Step 2.f: Call the Lagrange interpolator
  int err = interpolation_3d_general__uniform_src_grid(
      N_interp_GHOSTS,
      src_dxx0, src_dxx1, src_dxx2,
      Nxx_plus_2NGHOSTS0,
      Nxx_plus_2NGHOSTS1,
      Nxx_plus_2NGHOSTS2,
      num_interp_gfs,
      xx,
      src_gf_ptrs,
      num_interp_pts,
      dst_x0x1x2,
      dst_data);
  if (err != 0) {
    fprintf(stderr, "BHaH_get_metric_extrinsic_curvature: interpolation error %d\n", err);
    exit(1);
  }

  // Step 3: Unpack interpolated values at a single point
  REAL alphaL = dst_data[0][0];
  REAL cf = dst_data[1][0];
  REAL trK = dst_data[2][0];
  REAL vetU0 = dst_data[3][0];
  REAL vetU1 = dst_data[4][0];
  REAL vetU2 = dst_data[5][0];
  REAL hDD00 = dst_data[6][0];
  REAL hDD01 = dst_data[7][0];
  REAL hDD02 = dst_data[8][0];
  REAL hDD11 = dst_data[9][0];
  REAL hDD12 = dst_data[10][0];
  REAL hDD22 = dst_data[11][0];
  REAL aDD00 = dst_data[12][0];
  REAL aDD01 = dst_data[13][0];
  REAL aDD02 = dst_data[14][0];
  REAL aDD11 = dst_data[15][0];
  REAL aDD12 = dst_data[16][0];
  REAL aDD22 = dst_data[17][0];

  // Step 4: Compute ADM variables in Cartesian basis

  // Step 4.a: Lapse comes for free
  *alpha = alphaL;

  // Step 4.b: Compute shift betaCartU[3], extrinsic curvature KCartDD[3][3], and 3-metric gammaCartDD[3][3]
"""

    body += _generate_body_to_compute_ADM_variables(
        CoordSystem=CoordSystem, enable_rfm_precompute=False
    )

    cfc.register_CFunction(
        include_CodeParameters_h=False,  # it is already manually added in the function body
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        # CoordSystem_for_wrapper_func=CoordSystem,
        name=name,
        params=params,
        body=body,
        enable_simd=False,
    )
    return pcg.NRPyEnv()


if __name__ == "__main__":

    register_CFunction_BHaH_get_metric_extrinsic_curvature(
        CoordSystem="SinhCylindrical"
    )
    print(cfc.CFunction_dict["BHaH_get_metric_extrinsic_curvature"].full_function)
