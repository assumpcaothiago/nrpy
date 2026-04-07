"""Analytic-field fill and naive inner-BC pipeline for the diagnostic."""

import nrpy.c_function as cfc


def register_CFunction_counterexample2_fill_analytic_auxevol_fields() -> None:
    """Fill the analytic test field into AUXEVOL storage on the grid interior."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Fill the analytic field h_{theta theta} = r sin(theta) cos(phi) on the source-grid interior.",
        cfunc_type="void",
        name="counterexample2_fill_analytic_auxevol_fields",
        params="const commondata_struct *restrict commondata",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;

#pragma omp parallel for
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
    const REAL xx2 = commondata->interp_src_r_theta_phi[2][i2];
    const REAL cosphi = cos(xx2);
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
      const REAL xx1 = commondata->interp_src_r_theta_phi[1][i1];
      const REAL sintheta = sin(xx1);
      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
        const REAL xx0 = commondata->interp_src_r_theta_phi[0][i0];
        commondata->interp_src_gfs[IDX4(SRC_HDD11GF, i0, i1, i2)] = xx0 * sintheta * cosphi;
      }
    }
  }
""",
    )


def register_CFunction_counterexample2_apply_base_field_inner_bcs() -> None:
    """Apply the naive base-field parity fill at inner boundary points."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Apply the current naive parity-only inner-boundary fill to base fields used by the derivative pipeline.",
        cfunc_type="void",
        name="counterexample2_apply_base_field_inner_bcs",
        params="const commondata_struct *restrict commondata, const bc_struct *restrict interp_src_bcstruct",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;
  const bc_info_struct *restrict bc_info = &interp_src_bcstruct->bc_info;

#pragma omp parallel
  for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
    switch (which_gf) {
    case SRC_WWGF:
    case SRC_HDD00GF:
    case SRC_HDD01GF:
    case SRC_HDD02GF:
    case SRC_HDD11GF:
    case SRC_HDD12GF:
    case SRC_HDD22GF: {
#pragma omp for
      for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
        const int dstpt = interp_src_bcstruct->inner_bc_array[pt].dstpt;
        const int srcpt = interp_src_bcstruct->inner_bc_array[pt].srcpt;
        commondata->interp_src_gfs[IDX4pt(which_gf, dstpt)] =
            interp_src_bcstruct->inner_bc_array[pt].parity[interp_src_gf_parity[which_gf]] *
            commondata->interp_src_gfs[IDX4pt(which_gf, srcpt)];
      }
      break;
    }
    default:
      break;
    }
  }
""",
    )


def register_CFunction_counterexample2_apply_naive_derivative_inner_bcs() -> None:
    """Apply the current naive rank-3 parity fill to stored derivatives."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Apply the current naive parity-only inner-boundary fill to all stored derivative gridfunctions.",
        cfunc_type="void",
        name="counterexample2_apply_naive_derivative_inner_bcs",
        params="const commondata_struct *restrict commondata, const bc_struct *restrict interp_src_bcstruct",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;
  const bc_info_struct *restrict bc_info = &interp_src_bcstruct->bc_info;

#pragma omp parallel for collapse(2)
  for (int which_gf = 0; which_gf < NUM_INTERP_SRC_GFS; which_gf++) {
    for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
      const int dstpt = interp_src_bcstruct->inner_bc_array[pt].dstpt;
      const int srcpt = interp_src_bcstruct->inner_bc_array[pt].srcpt;
      commondata->interp_src_gfs[IDX4pt(which_gf, dstpt)] =
          interp_src_bcstruct->inner_bc_array[pt].parity[interp_src_gf_parity[which_gf]] *
          commondata->interp_src_gfs[IDX4pt(which_gf, srcpt)];
    }
  }
""",
    )


def register_CFunction_counterexample2_prepare_analytic_auxevol() -> None:
    """Run the full naive analytic-field derivative pipeline on the bound griddata storage."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Prepare the analytic field, inner boundary data, and stored derivatives on auxevol_gfs.",
        cfunc_type="int",
        name="counterexample2_prepare_analytic_auxevol",
        params="commondata_struct *restrict commondata, bc_struct *restrict interp_src_bcstruct",
        include_CodeParameters_h=False,
        body=r"""
  counterexample2_fill_analytic_auxevol_fields(commondata);

  const int error_code = bcstruct_set_up(commondata, commondata->interp_src_r_theta_phi, interp_src_bcstruct);
  if (error_code != BHAHAHA_SUCCESS)
    return error_code;

  counterexample2_apply_base_field_inner_bcs(commondata, interp_src_bcstruct);
  hDD_dD_and_W_dD_in_interp_src_grid_interior(commondata);
  apply_bcs_r_maxmin_partial_r_hDD_upwinding(commondata, commondata->interp_src_r_theta_phi, commondata->interp_src_gfs, false);
  counterexample2_apply_naive_derivative_inner_bcs(commondata, interp_src_bcstruct);

  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunctions() -> None:
    """Register all analytic pipeline helpers."""
    register_CFunction_counterexample2_fill_analytic_auxevol_fields()
    register_CFunction_counterexample2_apply_base_field_inner_bcs()
    register_CFunction_counterexample2_apply_naive_derivative_inner_bcs()
    register_CFunction_counterexample2_prepare_analytic_auxevol()
