# nrpy/infrastructures/manga/BHaH_set_PrimsAndPos_particles.py
"""
Generate function to cc.

Author: Thiago Assumpção
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_BHaH_set_PrimsAndPos_particles() -> (
    Union[None, pcg.NRPyEnv_type]
):
    """
    Register function to compute Tmunu from primitive variables.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h", "nanoflann_bridge.h"]

    desc = "Compute Tmunu from primitive variables."
    cfunc_type = "void"
    name = "BHaH_set_PrimsAndPos_particles"
    params = r"int nParticles, const REAL *prims_and_pos, BHaH_struct *bhahstruct"

    prefunc = """
#ifndef BHAH_DATA_RHO

// Primitives and particle position
#define BHAH_DATA_RHO 0
#define BHAH_DATA_VX 1
#define BHAH_DATA_VY 2
#define BHAH_DATA_VZ 3
#define BHAH_DATA_IE 4
#define BHAH_DATA_X 5
#define BHAH_DATA_Y 6
#define BHAH_DATA_Z 7

// Gradients d(q)/dx, d(q)/dy, d(q)/dz at particle position
#define BHAH_DATA_GRADRHO_X 8
#define BHAH_DATA_GRADRHO_Y 9
#define BHAH_DATA_GRADRHO_Z 10

#define BHAH_DATA_GRADVX_X 11
#define BHAH_DATA_GRADVX_Y 12
#define BHAH_DATA_GRADVX_Z 13

#define BHAH_DATA_GRADVY_X 14
#define BHAH_DATA_GRADVY_Y 15
#define BHAH_DATA_GRADVY_Z 16

#define BHAH_DATA_GRADVZ_X 17
#define BHAH_DATA_GRADVZ_Y 18
#define BHAH_DATA_GRADVZ_Z 19

#define BHAH_DATA_GRADIE_X 20
#define BHAH_DATA_GRADIE_Y 21
#define BHAH_DATA_GRADIE_Z 22

#define BHAH_DATA_COMPONENTS 23
#endif
"""

    body = r"""
  // Step 1: Unpack data from bhahstruct

  // Step 1.a: commondata & griddata
  commondata_struct *commondata = bhahstruct->commondata;
  griddata_struct *restrict griddata = bhahstruct->griddata;
  const int grid = 0; // single grid
  params_struct *restrict params = &griddata[grid].params;

  // Step 1.b: grid points
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];

  // Step 1.c: grid functions
  REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  REAL *restrict y_n = griddata[grid].gridfuncs.y_n_gfs;

#include "set_CodeParameters.h"

  // Step 2: Compute hydro grid bounds (MAXHYDROSIZE) and atmosphere values (rho_min and ie_min)

  // Step 2.a: Initialize dummy values
  REAL MAXHYDROSIZE = -1.0;
  REAL rho_min = 1.0e100;
  REAL ie_min = 1.0e100;

  // Step 2.b: Loop through all hydro particles to identify hydro grid size and minimum values of internal energy and density
#pragma omp parallel for reduction(max : MAXHYDROSIZE) reduction(min : rho_min, ie_min)
  for (int p = 0; p < nParticles; p++) {
    const int base_idx = p * BHAH_DATA_COMPONENTS;
    const REAL _rho = prims_and_pos[base_idx + BHAH_DATA_RHO];
    const REAL _ie = prims_and_pos[base_idx + BHAH_DATA_IE];
    const REAL xHydro = prims_and_pos[base_idx + BHAH_DATA_X];
    // Since hydro grid is a square box centered at the origin, we only need to check one dimension
    if (fabs(xHydro) > MAXHYDROSIZE)
      MAXHYDROSIZE = fabs(xHydro);
    if (_rho < rho_min)
      rho_min = _rho;
    if (_ie < ie_min)
      ie_min = _ie;
  } // END for (int p = 0; p < nParticles; p++)

  // Step 2.c: Consistency check
  if (MAXHYDROSIZE < 0.0) {
    printf("[BHaH_set_PrimsAndPos_particles] ERROR: MAXHYDROSIZE < 0.0. Something is wrong with particle positions.\n");
    exit(EXIT_FAILURE);
  }

  // Step 3: Unpack positions of each particle in Cartesian coordinates and create xyz array

  // Step 3.a: Allocate memory for array of Cartesian coordinates
  REAL *xyz = (REAL *)malloc((size_t)3 * (size_t)nParticles * sizeof(REAL));
  if (xyz == NULL) {
    printf("[BHaH_set_PrimsAndPos_particles] ERROR: failed to allocate xyz buffer.\n");
    exit(EXIT_FAILURE);
  }

  // Step 3.b: Populate array of Cartesian coordinates
  for (int p = 0; p < nParticles; p++) {
    const int base_idx = p * BHAH_DATA_COMPONENTS;
    xyz[3 * (size_t)p + 0] = (REAL)prims_and_pos[base_idx + BHAH_DATA_X];
    xyz[3 * (size_t)p + 1] = (REAL)prims_and_pos[base_idx + BHAH_DATA_Y];
    xyz[3 * (size_t)p + 2] = (REAL)prims_and_pos[base_idx + BHAH_DATA_Z];
  }

  // Step 4: Using the position of each particle, build a KD-tree
  BHaH_KDTree *kdtree = NULL;
  {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // KD-tree over packed xyz positions
    kdtree = BHaH_kdtree_build_3d_xyz(nParticles, xyz);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    const REAL elapsed_sec = (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("[Timing] BHaH_kdtree_build_3d_xyz: %.6f seconds\n", elapsed_sec);

    if (kdtree == NULL) {
      printf("[BHaH_set_PrimsAndPos_particles] ERROR: failed to build nanoflann KD-tree.\n");
      free(xyz);
      exit(EXIT_FAILURE);
    }
  }

  // Step 5: Loop through BHaH grid to populate Tmunu
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  LOOP_OMP("omp parallel for",
           i0, NGHOSTS, Nxx_plus_2NGHOSTS0 - NGHOSTS,
           i1, NGHOSTS, Nxx_plus_2NGHOSTS1 - NGHOSTS,
           i2, NGHOSTS, Nxx_plus_2NGHOSTS2 - NGHOSTS) {

    // Step 5.a.i: Set curvilinear coordinates
    const REAL xx0 = xx[0][i0];
    const REAL xx1 = xx[1][i1];
    const REAL xx2 = xx[2][i2];

    // Step 5.a.ii: Compute Cartesian coordinate of the grid point
    REAL xCart[3] = {0.0, 0.0, 0.0};
    const REAL xx012[3] = {xx0, xx1, xx2};
    xx_to_Cart(params, xx012, xCart);

    // Step 5.b: Declare variables for hydro particles (initialize to NANs to catch any errors)
    REAL rho = NAN;
    REAL vx = NAN;
    REAL vy = NAN;
    REAL vz = NAN;
    REAL ie = NAN;

    // Step 5.c: Set primitive values to atmosphere if outside hydro grid
    if ((fabs(xCart[0]) > MAXHYDROSIZE) || (fabs(xCart[1]) > MAXHYDROSIZE) || (fabs(xCart[2]) > MAXHYDROSIZE)) {
      rho = rho_min;
      vx = 0.0;
      vy = 0.0;
      vz = 0.0;
      ie = ie_min;
    }
    // Step 5.d: In case grid point is inside hydro grid, proceed to first-order reconstruction of primitives
    else {
      // Start with invalid particle index (pid) and nearest-neighbor distance (dist2)
      int pid = -1;
      REAL dist2 = NAN;

      // Step 5.e: Query KD-Tree for the nearest-neighbor particle of position xCart[3], and exit execution in case of error
      const int rc_nn = BHaH_kdtree_query_1nn_3d(kdtree, xCart, &pid, &dist2);
      if (rc_nn != 0 || pid < 0 || pid >= nParticles) {
        printf("[BHaH_set_PrimsAndPos_particles] ERROR: nearest particle at (x, y, z) = (%.8e, %.8e, %.8e) not found (return code rc_nn=%d)\n", xCart[0], xCart[1], xCart[2], rc_nn);
        exit(EXIT_FAILURE);
      }

      // Step 5.f: Extract primitive and related variables corresponding to the nearest particle
      const int base_idx = pid * BHAH_DATA_COMPONENTS;

      // Step 5.f.i: Primitive values at particle centre
      const REAL rho0 = prims_and_pos[base_idx + BHAH_DATA_RHO];
      const REAL vx0 = prims_and_pos[base_idx + BHAH_DATA_VX];
      const REAL vy0 = prims_and_pos[base_idx + BHAH_DATA_VY];
      const REAL vz0 = prims_and_pos[base_idx + BHAH_DATA_VZ];
      const REAL ie0 = prims_and_pos[base_idx + BHAH_DATA_IE];

      // Step 5.f.ii:  Particle position
      const REAL px = prims_and_pos[base_idx + BHAH_DATA_X];
      const REAL py = prims_and_pos[base_idx + BHAH_DATA_Y];
      const REAL pz = prims_and_pos[base_idx + BHAH_DATA_Z];

      // Step 5.f.iii: Offset from particle to gridpoint (in Cartesian coords)
      const REAL dx = xCart[0] - px;
      const REAL dy = xCart[1] - py;
      const REAL dz = xCart[2] - pz;

      // Step 5.f.iv: Gradients d(q)/dx^i at the particle
      const REAL gradrho_x = prims_and_pos[base_idx + BHAH_DATA_GRADRHO_X];
      const REAL gradrho_y = prims_and_pos[base_idx + BHAH_DATA_GRADRHO_Y];
      const REAL gradrho_z = prims_and_pos[base_idx + BHAH_DATA_GRADRHO_Z];

      const REAL gradvx_x = prims_and_pos[base_idx + BHAH_DATA_GRADVX_X];
      const REAL gradvx_y = prims_and_pos[base_idx + BHAH_DATA_GRADVX_Y];
      const REAL gradvx_z = prims_and_pos[base_idx + BHAH_DATA_GRADVX_Z];

      const REAL gradvy_x = prims_and_pos[base_idx + BHAH_DATA_GRADVY_X];
      const REAL gradvy_y = prims_and_pos[base_idx + BHAH_DATA_GRADVY_Y];
      const REAL gradvy_z = prims_and_pos[base_idx + BHAH_DATA_GRADVY_Z];

      const REAL gradvz_x = prims_and_pos[base_idx + BHAH_DATA_GRADVZ_X];
      const REAL gradvz_y = prims_and_pos[base_idx + BHAH_DATA_GRADVZ_Y];
      const REAL gradvz_z = prims_and_pos[base_idx + BHAH_DATA_GRADVZ_Z];

      const REAL gradie_x = prims_and_pos[base_idx + BHAH_DATA_GRADIE_X];
      const REAL gradie_y = prims_and_pos[base_idx + BHAH_DATA_GRADIE_Y];
      const REAL gradie_z = prims_and_pos[base_idx + BHAH_DATA_GRADIE_Z];

      // Step 5.f.v: First–order Taylor reconstruction at the gridpoint
      rho = rho0 + gradrho_x * dx + gradrho_y * dy + gradrho_z * dz;
      vx = vx0 + gradvx_x * dx + gradvx_y * dy + gradvx_z * dz;
      vy = vy0 + gradvy_x * dx + gradvy_y * dy + gradvy_z * dz;
      vz = vz0 + gradvz_x * dx + gradvz_y * dy + gradvz_z * dz;
      ie = ie0 + gradie_x * dx + gradie_y * dy + gradie_z * dz;
    } // END else after if ((fabs(xCart[0]) > MAXHYDROSIZE) || (fabs(xCart[1]) > MAXHYDROSIZE) || (fabs(xCart[2]) > MAXHYDROSIZE))

    // Step 6: Compute derived variables from primitives

    // Step 6.a: Compute corresponding pressure assuming Gamma-law EOS
    const REAL press = (commondata->poly_eos_Gamma - 1.0) * rho * ie;

    // Step 6.b: Compute rescaled 3-velocity in curvilinear coordinates
    const REAL vCartU[3] = {vx, vy, vz};
    REAL rescaledvU0, rescaledvU1, rescaledvU2;
    compute_rescaledvU_from_vCartU(commondata, params, vCartU, xx0, xx1, xx2,
                                   &rescaledvU0, &rescaledvU1, &rescaledvU2);

    // Step 6.c: Compute time component of 4-velocity and restrict 3-velocity
    REAL u4Ut;
    const REAL max_lorentz_factor = 10.0;
    compute_u4Ut(commondata, params, max_lorentz_factor, i0, i1, i2, y_n,
                 &rescaledvU0, &rescaledvU1, &rescaledvU2, &u4Ut);

    // Step 6.d: Compute specific enthalpy
    const REAL h = 1.0 + ie + press / rho;

    // Step 7: Point-wise computation of TUU[4][4] in curvilinear coordinates
    compute_T4UU(commondata, params, i0, i1, i2, xx, rho, press, h, u4Ut,
                 rescaledvU0, rescaledvU1, rescaledvU2, y_n, auxevol_gfs);

  } // END LOOP_OMP (over all grid points)

  clock_gettime(CLOCK_MONOTONIC, &t1);
  const REAL elapsed_sec = (t1.tv_sec - t0.tv_sec) + 1.0e-9 * (t1.tv_nsec - t0.tv_nsec);
  printf("[BHaH_set_PrimsAndPos_particles] First-order Tmunu reconstruction took: %.6f seconds\n", elapsed_sec);

  // Step 8: Free KD-tree and packed xyz buffer
  BHaH_kdtree_free(kdtree);
  free(xyz);
"""

    cfc.register_CFunction(
        include_CodeParameters_h=False,  # already manually included in function body
        prefunc=prefunc,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
    )
    return pcg.NRPyEnv()
