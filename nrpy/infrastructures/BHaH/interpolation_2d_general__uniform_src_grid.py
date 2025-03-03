"""
Register C function for 3D Lagrange interpolation at arbitrary sets of points.

Author: Zachariah B. Etienne; zachetie **at** gmail **dot* com
        Thiago Assumpcao (adapted to 2d)
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
from nrpy.helpers.generic import copy_files


def register_CFunction_interpolation_2d_general__uniform_src_grid(
    enable_simd: bool,
    project_dir: str,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register the C function for general-purpose 2D Lagrange interpolation.

    Even if `enable_simd` is False, `intrinsics/simd_intrinsics.h` is still required.

    DocTests:
    >>> env = register_CFunction_interpolation_2d_general__uniform_src_grid(enable_simd=False, project_dir=".")

    :param enable_simd: Whether the rest of the code enables SIMD optimizations, as this code requires simd_intrinsics.h (which includes SIMD-disabled options).
    :param project_dir: Directory of the project, to set the destination for simd_instrinsics.h .
    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    if not enable_simd:
        copy_files(
            package="nrpy.helpers",
            filenames_list=["simd_intrinsics.h"],
            project_dir=project_dir,
            subdirectory="intrinsics",
        )

    includes = ["stdio.h", "stdlib.h", "math.h", "intrinsics/simd_intrinsics.h"]

    prefunc = """
#ifndef REAL
#define REAL double
#endif

enum { INTERP_SUCCESS,
       INTERP2D_GENERAL_NULL_PTRS,
       INTERP2D_GENERAL_INTERP_ORDER_GT_NXX123,
       INTERP2D_GENERAL_HORIZON_OUT_OF_BOUNDS } error_codes;

#pragma GCC optimize("unroll-loops")
"""

    desc = r"""Performs 2D Lagrange interpolation (specialized from the 3D version) from a set of uniform grid points on the source grid to arbitrary destination points. 
The interpolation is performed in the x0 and x1 directions only; the x2 coordinate is fixed.

This function interpolates scalar grid functions from a source grid to a set of destination points in the x0 and x1 directions,
using Lagrange interpolation of order INTERP_ORDER. The x2 coordinate is fixed by using a single grid point.

@param N_interp_GHOSTS - Number of ghost zones from the center of source point; interpolation order = 2 * N_interp_GHOSTS + 1.
@param src_dxx0 - Grid spacing in the x0 direction on the source grid.
@param src_dxx1 - Grid spacing in the x1 direction on the source grid.
@param src_Nxx_plus_2NGHOSTS0 - Dimension of the source grid in x0, including ghost zones.
@param src_Nxx_plus_2NGHOSTS1 - Dimension of the source grid in x1, including ghost zones.
@param NUM_INTERP_GFS - Number of grid functions to interpolate.
@param src_x0x1x2 - Arrays of coordinate values for x0, x1, and x2 on the source grid.
@param src_gf_ptrs - Array of pointers to source grid functions data.
@param num_dst_pts - Number of destination points.
@param dst_x0x1 - Destination points' coordinates (x0, x1).
@param dst_data - Output interpolated data for each grid function at the destination points, of size [NUM_INTERP_GFS][num_dst_pts].

@return - Error code indicating success or type of error encountered.

@note - The function interpolates each grid function separately and stores the results independently.
The source and destination grids are assumed to be uniform in x0 and x1 directions.
The function assumes that the destination grid points are within the range of the source grid.
"""

    cfunc_type = "int"
    name = "interpolation_2d_general__uniform_src_grid"
    params = """
    const int N_interp_GHOSTS, const REAL src_dxx0, const REAL src_dxx1,
    const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1,
    const int NUM_INTERP_GFS, REAL *restrict src_x0x1x2[3],
    const REAL *restrict src_gf_ptrs[NUM_INTERP_GFS], const int num_dst_pts, const REAL dst_x0x1[][2],
    REAL *restrict dst_data[NUM_INTERP_GFS]"""

    body = r"""
  // Unpack parameters.
  const int NinterpGHOSTS = N_interp_GHOSTS;
  const int INTERP_ORDER = (2 * NinterpGHOSTS + 1); // Number of points in the stencil per dimension.

  const REAL src_invdxx0 = 1.0 / src_dxx0;
  const REAL src_invdxx1 = 1.0 / src_dxx1;
  // Note: src_dxx2 is not used because x2 is fixed.

  // Adjust normalization factor to account only for the 2D (x0 and x1) interpolation.
  const REAL src_invdxx01_INTERP_ORDERm1 = pow(src_dxx0 * src_dxx1, -(INTERP_ORDER - 1));

  // Check for null pointers in source coordinates and output data.
  if (src_x0x1x2[0] == NULL || src_x0x1x2[1] == NULL || src_x0x1x2[2] == NULL)
    return INTERP2D_GENERAL_NULL_PTRS;
  for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
    if (dst_data[gf] == NULL)
      return INTERP2D_GENERAL_NULL_PTRS;
  }

  // Ensure interpolation order does not exceed grid dimensions (x0 and x1 only).
  if (INTERP_ORDER > src_Nxx_plus_2NGHOSTS0 || INTERP_ORDER > src_Nxx_plus_2NGHOSTS1)
    return INTERP2D_GENERAL_INTERP_ORDER_GT_NXX123;

  // Precompute inverse denominators for Lagrange interpolation coefficients.
  REAL inv_denom[INTERP_ORDER];
  for (int i = 0; i < INTERP_ORDER; i++) {
    REAL denom = 1.0;
    for (int j = 0; j < i; j++)
      denom *= (REAL)(i - j);
    for (int j = i + 1; j < INTERP_ORDER; j++)
      denom *= (REAL)(i - j);
    inv_denom[i] = 1.0 / denom;
  }

  // For 2D interpolation, fix the x2 coordinate by choosing a constant index.
  // Here we choose fixed_idx_x2 = NinterpGHOSTS (this could be any valid index).
  const int fixed_idx_x2 = NinterpGHOSTS;
  const int base_idx_x2 = fixed_idx_x2; // x2 base index is fixed.

  // Perform interpolation for each destination point (only x0 and x1 vary).
  const REAL xxmin_incl_ghosts0 = src_x0x1x2[0][0];
  const REAL xxmin_incl_ghosts1 = src_x0x1x2[1][0];
  int error_flag = INTERP_SUCCESS;

#pragma omp parallel for
  for (int dst_pt = 0; dst_pt < num_dst_pts; dst_pt++) {
    // Extract destination point coordinates; ignore x2.
    const REAL x0_dst = dst_x0x1[dst_pt][0];
    const REAL x1_dst = dst_x0x1[dst_pt][1];
    int idx_center_x0 = (int)((x0_dst - xxmin_incl_ghosts0) * src_invdxx0 + 0.5);
    int idx_center_x1 = (int)((x1_dst - xxmin_incl_ghosts1) * src_invdxx1 + 0.5);
    // int idx_center_x2 = -1;  // x2 is not used.

    // Check if the stencil goes out of bounds in x0 or x1.
    if ((idx_center_x0 - NinterpGHOSTS < 0) || (idx_center_x0 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS0) ||
        (idx_center_x1 - NinterpGHOSTS < 0) || (idx_center_x1 + NinterpGHOSTS >= src_Nxx_plus_2NGHOSTS1)) {
#pragma omp critical
      {
        error_flag = INTERP2D_GENERAL_HORIZON_OUT_OF_BOUNDS;
        idx_center_x0 = idx_center_x1 = NinterpGHOSTS;
      }
      continue;
    }

    // Compute base indices for the x0 and x1 stencils.
    const int base_idx_x0 = idx_center_x0 - NinterpGHOSTS;
    const int base_idx_x1 = idx_center_x1 - NinterpGHOSTS;

    // Compute differences for Lagrange interpolation in x0 and x1.
    REAL coeff_x0[INTERP_ORDER], coeff_x1[INTERP_ORDER];
    REAL diffs_x0[INTERP_ORDER], diffs_x1[INTERP_ORDER];
    // For x2, we bypass any computation.
    // REAL coeff_x2[INTERP_ORDER]; // Only coeff_x2[0] will be used.

#pragma omp simd
    for (int j = 0; j < INTERP_ORDER; j++) {
      diffs_x0[j] = x0_dst - src_x0x1x2[0][base_idx_x0 + j];
      diffs_x1[j] = x1_dst - src_x0x1x2[1][base_idx_x1 + j];
      // No x2 difference is computed.
    }

    // Compute the Lagrange basis coefficients for x0 and x1.
#pragma omp simd
    for (int i = 0; i < INTERP_ORDER; i++) {
      REAL numer_i_x0 = 1.0, numer_i_x1 = 1.0;
      for (int j = 0; j < i; j++) {
        numer_i_x0 *= diffs_x0[j];
        numer_i_x1 *= diffs_x1[j];
      }
      for (int j = i + 1; j < INTERP_ORDER; j++) {
        numer_i_x0 *= diffs_x0[j];
        numer_i_x1 *= diffs_x1[j];
      }
      coeff_x0[i] = numer_i_x0 * inv_denom[i];
      coeff_x1[i] = numer_i_x1 * inv_denom[i];
      // For x2, force the coefficient to be 1 at i==0 and 0 otherwise.
      // coeff_x2[i] = (i == 0 ? 1.0 : 0.0);
    }

    // Combine the 1D coefficients in x0 and x1 to form a 2D coefficient array.
    REAL coeff_2d[INTERP_ORDER][INTERP_ORDER];
    for (int ix1 = 0; ix1 < INTERP_ORDER; ix1++) {
      for (int ix0 = 0; ix0 < INTERP_ORDER; ix0++) {
        coeff_2d[ix1][ix0] = coeff_x0[ix0] * coeff_x1[ix1];
      }
    }

    // For each grid function, compute the interpolated value.
    for (int gf = 0; gf < NUM_INTERP_GFS; gf++) {
      REAL sum = 0.0;
      // Note: x2 is fixed to base_idx_x2.
      for (int ix1 = 0; ix1 < INTERP_ORDER; ix1++) {
        const int idx1 = base_idx_x1 + ix1;
        int ix0 = 0;
        REAL_SIMD_ARRAY vec_sum = SetZeroSIMD;
        // Precompute the base offset for x0 and x1; use fixed base_idx_x2.
        const int base_offset = base_idx_x0 + src_Nxx_plus_2NGHOSTS0 * (idx1 + src_Nxx_plus_2NGHOSTS1 * base_idx_x2);
        for (; ix0 <= INTERP_ORDER - simd_width; ix0 += simd_width) {
          const int current_idx0 = base_offset + ix0;
          REAL_SIMD_ARRAY vec_src = ReadSIMD(&src_gf_ptrs[gf][current_idx0]);
          REAL_SIMD_ARRAY vec_coeff = ReadSIMD(&coeff_2d[ix1][ix0]);
          vec_sum = FusedMulAddSIMD(vec_src, vec_coeff, vec_sum);
        }
        sum += HorizAddSIMD(vec_sum);
        for (; ix0 < INTERP_ORDER; ix0++) {
          const int current_idx0 = base_offset + ix0;
          sum += src_gf_ptrs[gf][current_idx0] * coeff_2d[ix1][ix0];
        }
      }
      // Store the interpolated value, using the 2D normalization factor.
      dst_data[gf][dst_pt] = sum * src_invdxx01_INTERP_ORDERm1;
    }

  } // End parallel loop over destination points.

  return error_flag;
"""

    postfunc = (
        r"#pragma GCC reset_options // Reset compiler optimizations after the function"
    )

    cfc.register_CFunction(
        subdirectory="",
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        postfunc=postfunc,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())
