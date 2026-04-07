"""Cartesian sampling and CSV export for the Counterexample 2 diagnostic."""

import nrpy.c_function as cfc


def register_CFunction_counterexample2_validate_cartesian_domain() -> None:
    """Ensure the requested Cartesian box fits inside the spherical grid."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Validate that the spherical source grid extends far enough to cover the requested Cartesian box.",
        cfunc_type="int",
        name="counterexample2_validate_cartesian_domain",
        params="const commondata_struct *restrict commondata",
        include_CodeParameters_h=False,
        body=r"""
  const REAL eps_cart = COUNTEREXAMPLE2_CART_EPS_FACTOR * commondata->interp_src_dxx0;
  const REAL required_radius =
      sqrt(2.0 * COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH * COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH + eps_cart * eps_cart);
  const REAL available_radius =
      commondata->interp_src_r_theta_phi[0][NGHOSTS + commondata->interp_src_Nxx0 - 1] + 0.5 * commondata->interp_src_dxx0;
  if (required_radius > available_radius) {
    fprintf(stderr,
            "Counterexample 2 Cartesian sampling requires radius %.17e for half-width %.17e and eps %.17e, but only %.17e is available.\n",
            required_radius, (REAL)COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH, eps_cart, available_radius);
    return COUNTEREXAMPLE2_CARTESIAN_DOMAIN_TOO_SMALL;
  }
  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_set_cartesian_sample_point() -> None:
    """Implement the agreed deterministic singularity-avoidance offsets."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Apply the deterministic Cartesian epsilon-offset policy before converting points to spherical coordinates.",
        cfunc_type="void",
        name="counterexample2_set_cartesian_sample_point",
        params="""const commondata_struct *restrict commondata, const int sample_kind,
                const REAL x_req, const REAL y_req, const REAL z_req,
                REAL *restrict x_samp, REAL *restrict y_samp, REAL *restrict z_samp""",
        include_CodeParameters_h=False,
        body=r"""
  const REAL eps_cart = COUNTEREXAMPLE2_CART_EPS_FACTOR * commondata->interp_src_dxx0;

  *x_samp = x_req;
  *y_samp = y_req;
  *z_samp = z_req;

  switch (sample_kind) {
  case COUNTEREXAMPLE2_CART_X_LINE:
    if (fabs(x_req) < eps_cart)
      *x_samp = eps_cart;
    break;
  case COUNTEREXAMPLE2_CART_Y_LINE:
    *x_samp = (fabs(y_req) < eps_cart) ? eps_cart : 0.0;
    *y_samp = y_req;
    *z_samp = 0.0;
    break;
  case COUNTEREXAMPLE2_CART_Z_LINE:
    *x_samp = eps_cart;
    *y_samp = 0.0;
    *z_samp = z_req;
    break;
  case COUNTEREXAMPLE2_CART_XY_PLANE:
    if (fabs(x_req) < eps_cart && fabs(y_req) < eps_cart)
      *x_samp = eps_cart;
    *z_samp = 0.0;
    break;
  case COUNTEREXAMPLE2_CART_XZ_PLANE:
    if (fabs(x_req) < eps_cart)
      *x_samp = eps_cart;
    *y_samp = 0.0;
    break;
  case COUNTEREXAMPLE2_CART_YZ_PLANE:
    *x_samp = eps_cart;
    *y_samp = y_req;
    *z_samp = z_req;
    break;
  default:
    break;
  }
""",
    )


def register_CFunction_counterexample2_interpolate_cartesian_fields() -> None:
    """Directly sample the stored spherical partials at Cartesian locations."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Convert Cartesian samples to spherical coordinates and interpolate h_{theta theta,i} at those points.",
        cfunc_type="int",
        name="counterexample2_interpolate_cartesian_fields",
        params="""const commondata_struct *restrict commondata, const params_struct *restrict params, const int num_dst_pts,
                const REAL sampled_xyz[][3], REAL sampled_sph[][3],
                REAL *restrict hdd11_d0, REAL *restrict hdd11_d1, REAL *restrict hdd11_d2""",
        include_CodeParameters_h=False,
        body=r"""
  const int Nxx_plus_2NGHOSTS0 = commondata->interp_src_Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = commondata->interp_src_Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = commondata->interp_src_Nxx_plus_2NGHOSTS2;
  REAL *restrict *src_r_theta_phi = (REAL *restrict *)commondata->interp_src_r_theta_phi;
  const REAL *restrict src_gf_ptrs[COUNTEREXAMPLE2_NUM_CART_INTERP_GFS] = {
      &commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD011GF, 0, 0, 0)],
      &commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD111GF, 0, 0, 0)],
      &commondata->interp_src_gfs[IDX4(SRC_PARTIAL_D_HDD211GF, 0, 0, 0)],
  };
  REAL *restrict dst_data[COUNTEREXAMPLE2_NUM_CART_INTERP_GFS] = {hdd11_d0, hdd11_d1, hdd11_d2};

  for (int dst_pt = 0; dst_pt < num_dst_pts; dst_pt++) {
    const REAL xCart[3] = {sampled_xyz[dst_pt][0], sampled_xyz[dst_pt][1], sampled_xyz[dst_pt][2]};
    int nearest_i0i1i2[3];
    Cart_to_xx_and_nearest_i0i1i2(params, xCart, sampled_sph[dst_pt], nearest_i0i1i2);
  }

  const int raw_error_code = interpolation_3d_general__uniform_src_grid(
      NGHOSTS - 1, commondata->interp_src_dxx0, commondata->interp_src_dxx1, commondata->interp_src_dxx2,
      commondata->interp_src_Nxx_plus_2NGHOSTS0, commondata->interp_src_Nxx_plus_2NGHOSTS1, commondata->interp_src_Nxx_plus_2NGHOSTS2,
      COUNTEREXAMPLE2_NUM_CART_INTERP_GFS, src_r_theta_phi, src_gf_ptrs, num_dst_pts, sampled_sph, dst_data);
  if (raw_error_code != BHAHAHA_SUCCESS) {
    fprintf(stderr, "counterexample2_interpolate_cartesian_fields: interpolation_3d_general__uniform_src_grid returned %d\n", raw_error_code);
    return COUNTEREXAMPLE2_CARTESIAN_INTERP_FAILURE;
  }

  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_emit_cartesian_product() -> None:
    """Emit one Cartesian line or plane CSV."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Emit one Cartesian Counterexample 2 diagnostic product.",
        cfunc_type="int",
        name="counterexample2_emit_cartesian_product",
        params="""const commondata_struct *restrict commondata, const params_struct *restrict params, const int sample_kind,
                const char *restrict output_filename""",
        include_CodeParameters_h=False,
        body=r"""
  const int is_line = (sample_kind <= COUNTEREXAMPLE2_CART_Z_LINE);
  const int num_samples0 = is_line ? COUNTEREXAMPLE2_CART_LINE_SAMPLES : COUNTEREXAMPLE2_CART_PLANE_SAMPLES;
  const int num_samples1 = is_line ? 1 : COUNTEREXAMPLE2_CART_PLANE_SAMPLES;
  const int num_pts = num_samples0 * num_samples1;
  const REAL coord_min = -(REAL)COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH;
  const REAL coord_max = +(REAL)COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH;

  REAL(*requested_xyz)[3] = NULL;
  REAL(*sampled_xyz)[3] = NULL;
  REAL(*sampled_sph)[3] = NULL;
  REAL *restrict hdd11_d0 = NULL;
  REAL *restrict hdd11_d1 = NULL;
  REAL *restrict hdd11_d2 = NULL;

  BHAH_MALLOC(requested_xyz, sizeof(REAL[3]) * (size_t)num_pts);
  BHAH_MALLOC(sampled_xyz, sizeof(REAL[3]) * (size_t)num_pts);
  BHAH_MALLOC(sampled_sph, sizeof(REAL[3]) * (size_t)num_pts);
  BHAH_MALLOC(hdd11_d0, sizeof(REAL) * (size_t)num_pts);
  BHAH_MALLOC(hdd11_d1, sizeof(REAL) * (size_t)num_pts);
  BHAH_MALLOC(hdd11_d2, sizeof(REAL) * (size_t)num_pts);
  if (requested_xyz == NULL || sampled_xyz == NULL || sampled_sph == NULL || hdd11_d0 == NULL || hdd11_d1 == NULL || hdd11_d2 == NULL) {
    BHAH_FREE(requested_xyz);
    BHAH_FREE(sampled_xyz);
    BHAH_FREE(sampled_sph);
    BHAH_FREE(hdd11_d0);
    BHAH_FREE(hdd11_d1);
    BHAH_FREE(hdd11_d2);
    return COUNTEREXAMPLE2_TEMP_MALLOC_ERROR;
  }

  for (int i0 = 0; i0 < num_samples0; i0++) {
    const REAL lambda0 = (num_samples0 == 1) ? 0.0 : (REAL)i0 / (REAL)(num_samples0 - 1);
    const REAL coord0 = coord_min + lambda0 * (coord_max - coord_min);
    for (int i1 = 0; i1 < num_samples1; i1++) {
      const REAL lambda1 = (num_samples1 == 1) ? 0.0 : (REAL)i1 / (REAL)(num_samples1 - 1);
      const REAL coord1 = coord_min + lambda1 * (coord_max - coord_min);
      REAL x_req = 0.0;
      REAL y_req = 0.0;
      REAL z_req = 0.0;
      const int idx = i0 * num_samples1 + i1;

      switch (sample_kind) {
      case COUNTEREXAMPLE2_CART_X_LINE:
        x_req = coord0;
        break;
      case COUNTEREXAMPLE2_CART_Y_LINE:
        y_req = coord0;
        break;
      case COUNTEREXAMPLE2_CART_Z_LINE:
        z_req = coord0;
        break;
      case COUNTEREXAMPLE2_CART_XY_PLANE:
        x_req = coord0;
        y_req = coord1;
        break;
      case COUNTEREXAMPLE2_CART_XZ_PLANE:
        x_req = coord0;
        z_req = coord1;
        break;
      case COUNTEREXAMPLE2_CART_YZ_PLANE:
        y_req = coord0;
        z_req = coord1;
        break;
      default:
        BHAH_FREE(requested_xyz);
        BHAH_FREE(sampled_xyz);
        BHAH_FREE(sampled_sph);
        BHAH_FREE(hdd11_d0);
        BHAH_FREE(hdd11_d1);
        BHAH_FREE(hdd11_d2);
        return COUNTEREXAMPLE2_CARTESIAN_INTERP_FAILURE;
      }

      requested_xyz[idx][0] = x_req;
      requested_xyz[idx][1] = y_req;
      requested_xyz[idx][2] = z_req;
      counterexample2_set_cartesian_sample_point(commondata, sample_kind, x_req, y_req, z_req, &sampled_xyz[idx][0], &sampled_xyz[idx][1],
                                                 &sampled_xyz[idx][2]);
    }
  }

  int error_code = counterexample2_interpolate_cartesian_fields(commondata, params, num_pts, sampled_xyz, sampled_sph, hdd11_d0, hdd11_d1, hdd11_d2);
  if (error_code != BHAHAHA_SUCCESS) {
    BHAH_FREE(requested_xyz);
    BHAH_FREE(sampled_xyz);
    BHAH_FREE(sampled_sph);
    BHAH_FREE(hdd11_d0);
    BHAH_FREE(hdd11_d1);
    BHAH_FREE(hdd11_d2);
    return error_code;
  }

  FILE *restrict output_file = fopen(output_filename, "w");
  if (output_file == NULL) {
    BHAH_FREE(requested_xyz);
    BHAH_FREE(sampled_xyz);
    BHAH_FREE(sampled_sph);
    BHAH_FREE(hdd11_d0);
    BHAH_FREE(hdd11_d1);
    BHAH_FREE(hdd11_d2);
    return COUNTEREXAMPLE2_OUTPUT_IO_ERROR;
  }

  fprintf(output_file, "x_req,y_req,z_req,x_samp,y_samp,z_samp,r,theta,phi,hdd11_d0,hdd11_d1,hdd11_d2,dhdx,exact,abs_error\n");

  for (int idx = 0; idx < num_pts; idx++) {
    const REAL r = sampled_sph[idx][0];
    const REAL theta = sampled_sph[idx][1];
    const REAL phi = sampled_sph[idx][2];
    const REAL dhdx = counterexample2_compute_dhdx(r, theta, phi, hdd11_d0[idx], hdd11_d1[idx], hdd11_d2[idx]);
    const REAL abs_error = fabs(dhdx - 1.0);
    fprintf(output_file,
            "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,1.0,%.17e\n",
            requested_xyz[idx][0], requested_xyz[idx][1], requested_xyz[idx][2], sampled_xyz[idx][0], sampled_xyz[idx][1], sampled_xyz[idx][2], r,
            theta, phi, hdd11_d0[idx], hdd11_d1[idx], hdd11_d2[idx], dhdx, abs_error);
  }

  fclose(output_file);
  BHAH_FREE(requested_xyz);
  BHAH_FREE(sampled_xyz);
  BHAH_FREE(sampled_sph);
  BHAH_FREE(hdd11_d0);
  BHAH_FREE(hdd11_d1);
  BHAH_FREE(hdd11_d2);
  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_emit_cartesian_products() -> None:
    """Emit all Cartesian CSV products."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Emit all Cartesian Counterexample 2 diagnostic products.",
        cfunc_type="int",
        name="counterexample2_emit_cartesian_products",
        params="const commondata_struct *restrict commondata, const params_struct *restrict params, const char *restrict output_dir",
        include_CodeParameters_h=False,
        body=r"""
  int error_code = counterexample2_validate_cartesian_domain(commondata);
  if (error_code != BHAHAHA_SUCCESS)
    return error_code;

  const char *basenames[6] = {
      "cart_x_line.csv",
      "cart_y_line.csv",
      "cart_z_line.csv",
      "cart_xy_plane.csv",
      "cart_xz_plane.csv",
      "cart_yz_plane.csv",
  };
  char filenames[6][4096];

  for (int sample_kind = COUNTEREXAMPLE2_CART_X_LINE; sample_kind <= COUNTEREXAMPLE2_CART_YZ_PLANE; sample_kind++) {
    snprintf(filenames[sample_kind], sizeof(filenames[sample_kind]), "%s/%s", output_dir, basenames[sample_kind]);
    error_code = counterexample2_emit_cartesian_product(commondata, params, sample_kind, filenames[sample_kind]);
    if (error_code != BHAHAHA_SUCCESS)
      return error_code;
  }

  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunctions() -> None:
    """Register all Cartesian sampling and export helpers."""
    register_CFunction_counterexample2_validate_cartesian_domain()
    register_CFunction_counterexample2_set_cartesian_sample_point()
    register_CFunction_counterexample2_interpolate_cartesian_fields()
    register_CFunction_counterexample2_emit_cartesian_product()
    register_CFunction_counterexample2_emit_cartesian_products()
