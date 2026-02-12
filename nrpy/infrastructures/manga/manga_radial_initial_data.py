"""
Register function to generate radial initial data for MaNGa.

Authors: Thiago Assumpção
"""

import nrpy.c_function as cfc

def register_CFunction_manga_radial_initial_data() -> None:
    """
    Register C function manga_radial_initial_data().

    Set radial initial data quantities for MaNGa.
    """
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = "Set radial initial data quantities for MaNGa."
    prefunc = ""
    name = "manga_radial_initial_data"
    params = """commondata_struct *restrict commondata, griddata_struct *restrict griddata,
      const int num_radial_pts, REAL *restrict r_axis, REAL *restrict rho_baryon, REAL *restrict pressure"""
    body = r"""
  // Set params struct for a single grid
  const int grid = 0;
  params_struct *restrict params = &griddata[grid].params;

  // Declare ID struct and populate it with TOV solution
  ID_persist_struct ID_persist;
  TOVola_solve(commondata, &ID_persist);

  // Declare quantities for sampling local radial axis using SinhSpherical coordinates (this has nothing to do with the coordinates used by BHaH for evolution)
  const REAL domain_size = params->grid_physical_size;
  const REAL local_SINHW = 0.2;
  const REAL dx = 1.0 / ((REAL)num_radial_pts);

  // Populate arrays by interpolating rho_baryon and pressure onto r_axis
  for (int i = 0; i < num_radial_pts; i++) {
    const REAL x = dx / 2.0 + dx * i;
    r_axis[i] = domain_size * sinh(x / local_SINHW) / sinh(1.0 / local_SINHW);

  // Perform pointwise interpolation to radius r using ID_persist data
  REAL rho_energy_val, rho_baryon_val, P_val, M_val, expnu_val, exp4phi_val;

  TOVola_TOV_interpolate_1D(r_axis[i], commondata, commondata->max_interpolation_stencil_size, ID_persist.numpoints_arr, ID_persist.r_Schw_arr,
                            ID_persist.rho_energy_arr, ID_persist.rho_baryon_arr, ID_persist.P_arr, ID_persist.M_arr, ID_persist.expnu_arr,
                            ID_persist.exp4phi_arr, ID_persist.r_iso_arr, &rho_energy_val, &rho_baryon_val, &P_val, &M_val, &expnu_val,
                            &exp4phi_val);
  *rho_baryon = rho_baryon_val;
  *pressure = P_val;
  } // END for (int i = 0; i < num_radial_pts; i++)

  // Free memory allocated for ID struct
  {
    free(ID_persist.r_Schw_arr);
    free(ID_persist.rho_energy_arr);
    free(ID_persist.rho_baryon_arr);
    free(ID_persist.P_arr);
    free(ID_persist.M_arr);
    free(ID_persist.expnu_arr);
    free(ID_persist.exp4phi_arr);
    free(ID_persist.r_iso_arr);
  }
"""

    cfc.register_CFunction(
        subdirectory="BHAHLIB",
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        name=name,
        params=params,
        body=body,
    )
