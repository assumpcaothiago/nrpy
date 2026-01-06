"""
Register TOVola code TOVola_TOV_interpolate_1D.c.

TOVola creates Tolman-Oppenheimer-Volkoff spherically symmetric initial data,
 typically for single neutron stars.

Authors: David Boyer
         Zachariah B. Etienne
         Thiago Assumpção
"""

import nrpy.c_function as cfc


def register_CFunction_TOVola_TOV_interpolate_1D() -> None:
    """
    Register C function TOVola_TOV_interpolate_1D().

    Provides interpolation Function using Lagrange Polynomial.
    """
    includes = ["BHaH_defines.h"]
    desc = "Interpolation Function using Lagrange Polynomial."
    prefunc = r"""
/* Bisection index finder using binary search */
static int TOVola_bisection_idx_finder(const REAL rr_iso, const int numpoints_arr, const REAL *restrict r_iso_arr) {
  int x1 = 0;
  int x2 = numpoints_arr - 1;
  REAL y1 = rr_iso - r_iso_arr[x1];
  REAL y2 = rr_iso - r_iso_arr[x2];
  if (y1 * y2 > 0) {
    fprintf(stderr, "INTERPOLATION BRACKETING ERROR: r_iso_min = %e ?<= r_iso = %.15e ?<= %e = r_iso_max\n", r_iso_arr[0], rr_iso,
            r_iso_arr[numpoints_arr - 1]);
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < numpoints_arr; i++) {
    int x_midpoint = (x1 + x2) / 2;
    REAL y_midpoint = rr_iso - r_iso_arr[x_midpoint];
    if (y_midpoint * y1 <= 0) {
      x2 = x_midpoint;
      y2 = y_midpoint;
    } else {
      x1 = x_midpoint;
      y1 = y_midpoint;
    }
    if (abs(x2 - x1) == 1) {
      // If r_iso_arr[x1] is closer to rr_iso than r_iso_arr[x2] then return x1:
      if (fabs(rr_iso - r_iso_arr[x1]) < fabs(rr_iso - r_iso_arr[x2])) {
        return x1;
      }
      // Otherwise return x2:
      return x2;
    }
  }
  fprintf(stderr, "INTERPOLATION BRACKETING ERROR: r_iso_min = %e ?<= r_iso = %.15e ?<= %e = r_iso_max\n", r_iso_arr[0], rr_iso,
          r_iso_arr[numpoints_arr - 1]);
  exit(EXIT_FAILURE);
}
"""
    name = "TOVola_TOV_interpolate_1D"
    params = """REAL rr_iso, const commondata_struct *restrict commondata,
      const int interpolation_stencil_size, const int numpoints_arr, const REAL *restrict r_Schw_arr,
      const REAL *restrict rho_energy_arr, const REAL *restrict rho_baryon_arr, const REAL *restrict P_arr,
      const REAL *restrict M_arr, const REAL *restrict expnu_arr, const REAL *restrict exp4phi_arr,
      const REAL *restrict r_iso_arr, REAL *restrict rho_energy, REAL *restrict rho_baryon, REAL *restrict P,
      REAL *restrict M, REAL *restrict expnu, REAL *restrict exp4phi"""
    body = r"""
  const int R_idx = numpoints_arr - 1;
  const REAL M_star = M_arr[R_idx];
  const REAL r_iso_max_inside_star = r_iso_arr[R_idx];
  REAL r_Schw = 0.0;
  if (rr_iso < r_iso_max_inside_star) { // If we are INSIDE the star, we need to interpollate the data to the grid.
    // For this case, we know that for all our scalars, f(r) = f(-r)
    if (rr_iso < 0)
      rr_iso = -rr_iso;

    // First find the central interpolation stencil index:
    int idx_mid = TOVola_bisection_idx_finder(rr_iso, numpoints_arr, r_iso_arr);

    /* Use standard library functions instead of redefining macros */
    int idxmin = MAX(0, idx_mid - commondata->interpolation_stencil_size / 2 - 1);

    // -= Do not allow the interpolation stencil to cross the star's surface =-
    // max index is when idxmin + (commondata->interpolation_stencil_size-1) = R_idx
    //  -> idxmin at most can be R_idx - commondata->interpolation_stencil_size + 1
    idxmin = MIN(idxmin, R_idx - commondata->interpolation_stencil_size + 1);

    // Ensure that commondata->interpolation_stencil_size does not exceed the maximum
    if (commondata->interpolation_stencil_size > commondata->max_interpolation_stencil_size) {
      fprintf(stderr, "Interpolation stencil size exceeds maximum allowed.\n");
      exit(EXIT_FAILURE);
    }

    // Now perform the Lagrange polynomial interpolation:

    // First compute the interpolation coefficients:
    REAL r_iso_sample[commondata->max_interpolation_stencil_size];
    for (int i = idxmin; i < idxmin + commondata->interpolation_stencil_size; i++) {
      //if(i < 0 || i >= R_idx-1) { fprintf(stderr, "ERROR!\n"); exit(1); }
      r_iso_sample[i - idxmin] = r_iso_arr[i];
    }
    REAL l_i_of_r[commondata->max_interpolation_stencil_size];
    for (int i = 0; i < commondata->interpolation_stencil_size; i++) {
      REAL numer = 1.0;
      REAL denom = 1.0;
      for (int j = 0; j < commondata->interpolation_stencil_size; j++) {
        if (j != i) {
          numer *= (rr_iso - r_iso_sample[j]);
          denom *= (r_iso_sample[i] - r_iso_sample[j]);
        }
      }
      l_i_of_r[i] = numer / denom;
    }

    // Then perform the interpolation:
    *rho_energy = 0.0;
    *rho_baryon = 0.0;
    *P = 0.0;
    *M = 0.0;
    *expnu = 0.0;
    *exp4phi = 0.0;

    for (int i = idxmin; i < idxmin + commondata->interpolation_stencil_size; i++) {
      r_Schw += l_i_of_r[i - idxmin] * r_Schw_arr[i];
      *rho_energy += l_i_of_r[i - idxmin] * rho_energy_arr[i];
      *rho_baryon += l_i_of_r[i - idxmin] * rho_baryon_arr[i];
      *P += l_i_of_r[i - idxmin] * P_arr[i];
      *M += l_i_of_r[i - idxmin] * M_arr[i];
      *expnu += l_i_of_r[i - idxmin] * expnu_arr[i];
      *exp4phi += l_i_of_r[i - idxmin] * exp4phi_arr[i];
    }

  } else {
    // If we are OUTSIDE the star, the solution is just Schwarzschild.
    r_Schw = (rr_iso + M_star) + M_star * M_star / (4.0 * rr_iso); // Need to know what r_Schw is at our current grid location.
    *rho_energy = 0;
    *rho_baryon = 0;
    *P = 0;
    *M = M_star;
    *expnu = 1. - 2.0 * (M_star) / r_Schw;
    *exp4phi = (r_Schw * r_Schw) / (rr_iso * rr_iso);
  }
  //printf("%.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e hhhh\n", rr_iso, r_Schw, *rho_energy, *rho_baryon, *P, *M, *expnu, *exp4phi);
"""

    cfc.register_CFunction(
        subdirectory="TOVola",
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        name=name,
        params=params,
        body=body,
    )
