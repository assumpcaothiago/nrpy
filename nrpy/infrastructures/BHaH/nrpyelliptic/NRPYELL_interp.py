"""
Register NRPyElliptic code NRPYELL_interp.c.

Author: Thiago Assumpção; assumpcaothiago **at** gmail **dot** com
"""

import nrpy.c_function as cfc


def ID_persist_str() -> str:
    """
    Return contents of ID_persist_struct for NRPYELL initial data.

    :return: ID_persist_struct contents.
    """
    return r"""
  // NRPyElliptic initial data quantities
  int NRPYELL_Nxx_plus_2NGHOSTS0;
  int NRPYELL_Nxx_plus_2NGHOSTS1;
  int NRPYELL_Nxx_plus_2NGHOSTS2;
  int NRPYELL_NGHOSTS, NRPYELL_TOTAL_PTS;
  REAL NRPYELL_AMAX;
  REAL NRPYELL_bScale;
  REAL NRPYELL_SINHWAA;
  REAL NRPYELL_dxx0;
  REAL NRPYELL_dxx1;
  REAL NRPYELL_dxx2;
  REAL *restrict NRPYELL_xx0;
  REAL *restrict NRPYELL_xx1;
  REAL *restrict NRPYELL_xx2;
  REAL *restrict NRPYELL_rho;
  REAL *restrict NRPYELL_P;
  REAL *restrict NRPYELL_psi_minus_one;
  REAL *restrict NRPYELL_alphaconf_minus_one;
"""


def register_CFunction_NRPYELL_interp() -> None:
    """
    Register C function NRPYELL_interp().

    Provides interpolator to compute data at arbitrary point x,y,z in Spherical basis.
    """
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = "Provide high-order interpolation from NRPYELL grids onto an arbitrary point xCart[3] = {x,y,z} in the Spherical basis."
    prefunc = ""
    name = "NRPYELL_interp"
    params = """const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xCart[3],
      const ID_persist_struct *restrict ID_persist, initial_data_struct *restrict initial_data"""
    body = r"""
  // Load grid points from NRPyElliptic solver
  REAL *restrict NRPYELL_xx[3];
	NRPYELL_xx[0] = ID_persist->NRPYELL_xx0;
	NRPYELL_xx[1] = ID_persist->NRPYELL_xx1;
	NRPYELL_xx[2] = ID_persist->NRPYELL_xx2;

  // Compute reference metric coordinates from Cartesian coordinates
	REAL xx[3];
	NRPYELL_Cart_to_xx__rfm__SinhSymTP(ID_persist->NRPYELL_AMAX, ID_persist->NRPYELL_bScale, ID_persist->NRPYELL_SINHWAA, xCart, xx);

	// Set up destination grid point in a format expected by the interpolation function
	const REAL dst_x0x1[1][2] = {{xx[0], xx[1]}};

	// Set up pointers to NRPyElliptic data for function to be interpolated
	const REAL *src_gf_ptrs[4] = {ID_persist->NRPYELL_psi_minus_one, ID_persist->NRPYELL_alphaconf_minus_one,
																ID_persist->NRPYELL_rho, ID_persist->NRPYELL_P};

	// Set up variables for the interpolated quantities
	REAL psi_minus_one, alphaconf_minus_one, rho, P;
	REAL *dst_data[4] = {&psi_minus_one, &alphaconf_minus_one, &rho, &P};

	// Interpolate solution at a single point
	const int NUM_INTERP_GFS = 4;
	const int num_dst_pts = 1;
  int error_flag = interpolation_2d_general__uniform_src_grid(ID_persist->NRPYELL_NGHOSTS, ID_persist->NRPYELL_dxx0, ID_persist->NRPYELL_dxx1,
                                             ID_persist->NRPYELL_Nxx_plus_2NGHOSTS0, ID_persist->NRPYELL_Nxx_plus_2NGHOSTS1,
                                             NUM_INTERP_GFS, NRPYELL_xx, src_gf_ptrs, num_dst_pts, dst_x0x1, dst_data);
  if (error_flag){
		perror("NRPYELL interpolation error\n");
		exit(1);
	}

	// Computer psi and alpha from interpolated quantities
	const REAL psi = 1.0 + psi_minus_one;
	const REAL alpha = (1.0 + alphaconf_minus_one) / psi;

  // Set Cartesian coordinates
  const REAL Cartx = xCart[0];
  const REAL Carty = xCart[1];
  const REAL Cartz = xCart[2];

	// Set local isotropic radial coordinate
	const REAL r_iso = sqrt(Cartx * Cartx + Carty * Carty + Cartz * Cartz);
	const REAL theta = acos(Cartz / r_iso);

	const REAL NRPYELL_psi4 = psi * psi * psi * psi;
	const REAL NRPYELL_alpha2 = alpha * alpha;
	// const REAL rho_energy_val = rho;
	// const REAL P_val = P;

  /**
   *  NOTE: Instead of using rho and P interpolated from NRPYELL, we will use
   *        rho_energy_val and P_val from the TOVola solver. This avoids interpolation
   *        across stellar surfaces. To accomplish this, we compute `r_star`, the isotropic
   *        radius centered at each TOV star.
   *
   *  TODO: read `zpos` from NRPYLL_solution.bin (which requires modifying the reader as well)
   *        and get rid of hard-coded 5.0 below
   */
  REAL r_star = -1.0;
  if (Cartz > 0.0){
    r_star = sqrt(Cartx * Cartx + Carty * Carty + (Cartz - 5.0) * (Cartz - 5.0));
  }
  else{
    r_star = sqrt(Cartx * Cartx + Carty * Carty + (Cartz + 5.0) * (Cartz + 5.0));
  }

  // Perform pointwise interpolation to radius r using ID_persist data
  REAL rho_energy_val, rho_baryon_val, P_val, M_val, expnu_val, exp4phi_val;
  TOVola_TOV_interpolate_1D(r_star, commondata, commondata->max_interpolation_stencil_size, ID_persist->numpoints_arr, ID_persist->r_Schw_arr,
                            ID_persist->rho_energy_arr, ID_persist->rho_baryon_arr, ID_persist->P_arr, ID_persist->M_arr, ID_persist->expnu_arr,
                            ID_persist->exp4phi_arr, ID_persist->r_iso_arr, &rho_energy_val, &rho_baryon_val, &P_val, &M_val, &expnu_val,
                            &exp4phi_val);

	// printf("xCart, yCart, zCart, psi, alpha = %.4e, %.4e, %.4e, %.4e %.4e\n", xCart[0], xCart[1], xCart[2], psi, alpha);

  // Assign interpolated values to initial_data_struct
  initial_data->alpha = alpha;

  // Assuming beta and B fields are zero in this context
  initial_data->betaSphorCartU0 = 0.0;
  initial_data->betaSphorCartU1 = 0.0;
  initial_data->betaSphorCartU2 = 0.0;
  initial_data->BSphorCartU0 = 0.0;
  initial_data->BSphorCartU1 = 0.0;
  initial_data->BSphorCartU2 = 0.0;

  // Metric components (assuming diagonal for simplicity)
  initial_data->gammaSphorCartDD00 = NRPYELL_psi4;
  initial_data->gammaSphorCartDD01 = 0.0;
  initial_data->gammaSphorCartDD02 = 0.0;
  initial_data->gammaSphorCartDD11 = NRPYELL_psi4 * r_iso * r_iso;
  initial_data->gammaSphorCartDD12 = 0.0;
  initial_data->gammaSphorCartDD22 = NRPYELL_psi4 * r_iso * r_iso * sin(theta) * sin(theta);

  // Extrinsic curvature components set to zero
  initial_data->KSphorCartDD00 = 0.0;
  initial_data->KSphorCartDD01 = 0.0;
  initial_data->KSphorCartDD02 = 0.0;
  initial_data->KSphorCartDD11 = 0.0;
  initial_data->KSphorCartDD12 = 0.0;
  initial_data->KSphorCartDD22 = 0.0;

  initial_data->T4SphorCartUU00 = rho_energy_val / NRPYELL_alpha2;
  initial_data->T4SphorCartUU01 = 0.0;
  initial_data->T4SphorCartUU02 = 0.0;
  initial_data->T4SphorCartUU03 = 0.0;
  initial_data->T4SphorCartUU11 = P_val / NRPYELL_psi4;
  initial_data->T4SphorCartUU12 = 0.0;
  initial_data->T4SphorCartUU13 = 0.0;
  initial_data->T4SphorCartUU22 = P_val / (NRPYELL_psi4 * r_iso * r_iso);
  initial_data->T4SphorCartUU23 = 0.0;
  initial_data->T4SphorCartUU33 = P_val / (NRPYELL_psi4 * r_iso * r_iso * sin(theta) * sin(theta));
"""

    cfc.register_CFunction(
        subdirectory="NRPYELL",
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        name=name,
        params=params,
        body=body,
    )
