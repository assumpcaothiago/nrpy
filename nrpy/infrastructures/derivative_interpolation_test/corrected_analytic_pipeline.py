"""Corrected analytic-field fill and Jacobian-aware inner-BC pipeline."""

from nrpy.infrastructures import BHaH
import nrpy.c_function as cfc


def register_error_codes() -> None:
    """Register corrected-pipeline-specific error codes with BHaHAHA's error handler."""
    error_tuple = (
        "COUNTEREXAMPLE2_CORRECTED_UNKNOWN_INNER_MAP",
        "Counterexample 2 corrected diagnostic: could not classify the spherical inner-boundary map.",
    )
    existing = {
        name for name, _ in BHaH.BHaHAHA.error_message.error_code_msg_tuples_list
    }
    if error_tuple[0] not in existing:
        BHaH.BHaHAHA.error_message.error_code_msg_tuples_list.append(error_tuple)


def register_CFunction_counterexample2_corrected_wrap_to_pi() -> None:
    """Register helper wrapping an angle into [-pi, pi)."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Wrap an angle into the principal interval [-pi, pi).",
        cfunc_type="REAL",
        name="counterexample2_corrected_wrap_to_pi",
        params="const REAL angle",
        include_CodeParameters_h=False,
        body=r"""
  REAL wrapped = fmod(angle + M_PI, 2.0 * M_PI);
  if (wrapped < 0.0)
    wrapped += 2.0 * M_PI;
  return wrapped - M_PI;
""",
    )


def register_CFunction_counterexample2_corrected_decode_idx3() -> None:
    """Register helper decoding IDX3-packed indices."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Decode an IDX3-packed grid index into (i0, i1, i2).",
        cfunc_type="void",
        name="counterexample2_corrected_decode_idx3",
        params="""const commondata_struct *restrict commondata, const int pt,
                int idx3[3]""",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;

  idx3[0] = pt % Nxx_plus_2NGHOSTS0;
  const int tmp = pt / Nxx_plus_2NGHOSTS0;
  idx3[1] = tmp % Nxx_plus_2NGHOSTS1;
  idx3[2] = tmp / Nxx_plus_2NGHOSTS1;
""",
    )


def register_CFunction_counterexample2_corrected_set_inner_map_jacobian() -> None:
    """Register helper computing Jacobian signs for the effective spherical symmetry map."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Infer the diagonal Jacobian signs for the effective spherical symmetry map from one inner-boundary point pair.",
        cfunc_type="int",
        name="counterexample2_corrected_set_inner_map_jacobian",
        params="""const commondata_struct *restrict commondata,
                const REAL dst_r, const REAL dst_theta, const REAL dst_phi,
                const REAL src_r, const REAL src_theta, const REAL src_phi,
                int jac_sign[3]""",
        include_CodeParameters_h=False,
        body=r"""
  const REAL eps_r = 1.0e-12 + 0.25 * commondata->interp_src_dxx0;
  const REAL eps_ang = 1.0e-12 +
                       0.25 * ((commondata->interp_src_dxx1 > commondata->interp_src_dxx2) ? commondata->interp_src_dxx1
                                                                                             : commondata->interp_src_dxx2);
  const REAL phi_shift = counterexample2_corrected_wrap_to_pi(src_phi - dst_phi);

  if (fabs(src_r - dst_r) <= eps_r)
    jac_sign[0] = +1;
  else if (fabs(src_r + dst_r) <= eps_r)
    jac_sign[0] = -1;
  else {
    fprintf(stderr,
            "Counterexample 2 corrected BC: could not infer J^r_r from dst=(%.17e, %.17e, %.17e), src=(%.17e, %.17e, %.17e)\n",
            dst_r, dst_theta, dst_phi, src_r, src_theta, src_phi);
    return COUNTEREXAMPLE2_CORRECTED_UNKNOWN_INNER_MAP;
  }

  if (fabs(src_theta - dst_theta) <= eps_ang || fabs(src_theta - (dst_theta + M_PI)) <= eps_ang ||
      fabs(src_theta - (dst_theta - M_PI)) <= eps_ang)
    jac_sign[1] = +1;
  else if (fabs(src_theta + dst_theta) <= eps_ang || fabs(src_theta - (M_PI - dst_theta)) <= eps_ang ||
           fabs(src_theta - (2.0 * M_PI - dst_theta)) <= eps_ang)
    jac_sign[1] = -1;
  else {
    fprintf(stderr,
            "Counterexample 2 corrected BC: could not infer J^theta_theta from dst=(%.17e, %.17e, %.17e), src=(%.17e, %.17e, %.17e)\n",
            dst_r, dst_theta, dst_phi, src_r, src_theta, src_phi);
    return COUNTEREXAMPLE2_CORRECTED_UNKNOWN_INNER_MAP;
  }

  if (!(fabs(phi_shift) <= eps_ang || fabs(fabs(phi_shift) - M_PI) <= eps_ang)) {
    fprintf(stderr,
            "Counterexample 2 corrected BC: unexpected phi shift %.17e for dst=(%.17e, %.17e, %.17e), src=(%.17e, %.17e, %.17e)\n",
            phi_shift, dst_r, dst_theta, dst_phi, src_r, src_theta, src_phi);
    return COUNTEREXAMPLE2_CORRECTED_UNKNOWN_INNER_MAP;
  }

  jac_sign[2] = +1;
  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_corrected_apply_inner_bc_for_stored_derivs() -> None:
    """Register helper applying the corrected inner BC to stored derivative fields."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Apply the Jacobian-aware inner-boundary fill for stored coordinate derivatives on the interpolation source grid.",
        cfunc_type="int",
        name="counterexample2_corrected_apply_inner_bc_for_stored_derivs",
        params="""const commondata_struct *restrict commondata,
                const bc_struct *restrict interp_src_bcstruct""",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;
  const int deriv_gfs[21] = {
      SRC_PARTIAL_D_WW0GF,   SRC_PARTIAL_D_WW1GF,   SRC_PARTIAL_D_WW2GF,   SRC_PARTIAL_D_HDD000GF, SRC_PARTIAL_D_HDD001GF,
      SRC_PARTIAL_D_HDD002GF, SRC_PARTIAL_D_HDD011GF, SRC_PARTIAL_D_HDD012GF, SRC_PARTIAL_D_HDD022GF, SRC_PARTIAL_D_HDD100GF,
      SRC_PARTIAL_D_HDD101GF, SRC_PARTIAL_D_HDD102GF, SRC_PARTIAL_D_HDD111GF, SRC_PARTIAL_D_HDD112GF, SRC_PARTIAL_D_HDD122GF,
      SRC_PARTIAL_D_HDD200GF, SRC_PARTIAL_D_HDD201GF, SRC_PARTIAL_D_HDD202GF, SRC_PARTIAL_D_HDD211GF, SRC_PARTIAL_D_HDD212GF,
      SRC_PARTIAL_D_HDD222GF,
  };
  const int deriv_dir[21] = {
      0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
  };
  const int base_parity_type[21] = {
      0, 0, 0, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9,
  };
  const bc_info_struct *restrict bc_info = &interp_src_bcstruct->bc_info;

  for (int pt = 0; pt < bc_info->num_inner_boundary_points; pt++) {
    const int dstpt = interp_src_bcstruct->inner_bc_array[pt].dstpt;
    const int srcpt = interp_src_bcstruct->inner_bc_array[pt].srcpt;
    int dst_idx3[3];
    int src_idx3[3];
    int jac_sign[3];

    counterexample2_corrected_decode_idx3(commondata, dstpt, dst_idx3);
    counterexample2_corrected_decode_idx3(commondata, srcpt, src_idx3);

    const REAL dst_r = commondata->interp_src_r_theta_phi[0][dst_idx3[0]];
    const REAL dst_theta = commondata->interp_src_r_theta_phi[1][dst_idx3[1]];
    const REAL dst_phi = commondata->interp_src_r_theta_phi[2][dst_idx3[2]];
    const REAL src_r = commondata->interp_src_r_theta_phi[0][src_idx3[0]];
    const REAL src_theta = commondata->interp_src_r_theta_phi[1][src_idx3[1]];
    const REAL src_phi = commondata->interp_src_r_theta_phi[2][src_idx3[2]];

    const int error_code =
        counterexample2_corrected_set_inner_map_jacobian(commondata, dst_r, dst_theta, dst_phi, src_r, src_theta, src_phi, jac_sign);
    if (error_code != BHAHAHA_SUCCESS)
      return error_code;

    for (int entry = 0; entry < 21; entry++) {
      const int which_gf = deriv_gfs[entry];
      const int sign =
          ((base_parity_type[entry] == 0) ? 1 : interp_src_bcstruct->inner_bc_array[pt].parity[base_parity_type[entry]]) * jac_sign[deriv_dir[entry]];
      ((REAL *restrict)commondata->interp_src_gfs)[IDX4pt(which_gf, dstpt)] =
          sign * ((REAL *restrict)commondata->interp_src_gfs)[IDX4pt(which_gf, srcpt)];
    }
  }

  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_prepare_analytic_auxevol() -> None:
    """Run the full corrected analytic-field derivative pipeline on the bound griddata storage."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Prepare the analytic field, inner boundary data, and stored derivatives on auxevol_gfs using the corrected derivative inner BC.",
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
  return counterexample2_corrected_apply_inner_bc_for_stored_derivs(commondata, interp_src_bcstruct);
""",
    )


def register_CFunctions() -> None:
    """Register all corrected analytic-pipeline helpers."""
    register_CFunction_counterexample2_corrected_wrap_to_pi()
    register_CFunction_counterexample2_corrected_decode_idx3()
    register_CFunction_counterexample2_corrected_set_inner_map_jacobian()
    register_CFunction_counterexample2_corrected_apply_inner_bc_for_stored_derivs()
    register_CFunction_counterexample2_prepare_analytic_auxevol()
