"""Custom driver for the Counterexample 2 minimal diagnostic."""

import nrpy.c_function as cfc


def register_CFunction_counterexample2_ensure_directory_exists() -> None:
    """Create the requested output directory if it does not already exist."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "sys/stat.h", "sys/types.h"],
        desc="Ensure that the requested output directory exists, creating intermediate directories as needed.",
        cfunc_type="int",
        name="counterexample2_ensure_directory_exists",
        params="const char *restrict output_dir",
        include_CodeParameters_h=False,
        body=r"""
  if (output_dir == NULL || output_dir[0] == '\0')
    return COUNTEREXAMPLE2_OUTPUT_IO_ERROR;

  char path_copy[4096];
  const size_t path_len = strlen(output_dir);
  if (path_len >= sizeof(path_copy))
    return COUNTEREXAMPLE2_OUTPUT_IO_ERROR;

  snprintf(path_copy, sizeof(path_copy), "%s", output_dir);
  for (char *p = path_copy + 1; *p != '\0'; p++) {
    if (*p != '/')
      continue;
    *p = '\0';
    if (mkdir(path_copy, 0777) != 0 && errno != EEXIST)
      return COUNTEREXAMPLE2_OUTPUT_IO_ERROR;
    *p = '/';
  }

  if (mkdir(path_copy, 0777) != 0 && errno != EEXIST) {
    struct stat path_stat;
    if (stat(path_copy, &path_stat) != 0 || !S_ISDIR(path_stat.st_mode))
      return COUNTEREXAMPLE2_OUTPUT_IO_ERROR;
  }

  return BHAHAHA_SUCCESS;
""",
    )


def register_CFunction_counterexample2_cleanup() -> None:
    """Free all locally owned allocations in the custom diagnostic driver."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Free all locally owned allocations made by the Counterexample 2 diagnostic driver.",
        cfunc_type="void",
        name="counterexample2_cleanup",
        params="""commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                bc_struct *restrict interp_src_bcstruct""",
        include_CodeParameters_h=False,
        body=r"""
  if (interp_src_bcstruct != NULL) {
    BHAH_FREE(interp_src_bcstruct->inner_bc_array);
    for (int ng = 0; ng < NGHOSTS * 3; ng++) {
      BHAH_FREE(interp_src_bcstruct->pure_outer_bc_array[ng]);
    }
  }

  if (griddata != NULL) {
    BHAH_FREE(griddata[0].gridfuncs.auxevol_gfs);
    for (int dirn = 0; dirn < 3; dirn++) {
      BHAH_FREE(griddata[0].xx[dirn]);
      commondata->interp_src_r_theta_phi[dirn] = NULL;
    }
  }
  commondata->interp_src_gfs = NULL;
""",
    )


def register_CFunction_counterexample2_run() -> None:
    """Run the complete diagnostic workflow."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h", "string.h"],
        desc="Run the Counterexample 2 derivative diagnostic and export Cartesian CSV products.",
        cfunc_type="int",
        name="counterexample2_run",
        params="const char *restrict output_dir",
        include_CodeParameters_h=False,
        body=r"""
  int error_code = BHAHAHA_SUCCESS;
  commondata_struct commondata;
  griddata_struct *restrict griddata = NULL;
  bc_struct interp_src_bcstruct;
  bhahaha_params_and_data_struct diagnostic_params;

  memset(&commondata, 0, sizeof(commondata));
  memset(&interp_src_bcstruct, 0, sizeof(interp_src_bcstruct));
  memset(&diagnostic_params, 0, sizeof(diagnostic_params));

  commondata_struct_set_to_default(&commondata);
  commondata.NUMGRIDS = 1;

  BHAH_MALLOC(griddata, sizeof(griddata_struct));
  if (griddata == NULL)
    return COUNTEREXAMPLE2_GRIDDATA_MALLOC_ERROR;
  memset(griddata, 0, sizeof(griddata_struct));

  params_struct_set_to_default(&commondata, griddata);
  diagnostic_params.r_min_external_input = 0.0;
  commondata.bhahaha_params_and_data = &diagnostic_params;

  numerical_grids_and_timestep(&commondata, griddata, true);
  error_code = counterexample2_allocate_auxevol_gfs(griddata);
  if (error_code == BHAHAHA_SUCCESS)
    counterexample2_bind_griddata_to_interp_src(&commondata, griddata);
  if (error_code == BHAHAHA_SUCCESS)
    error_code = counterexample2_prepare_analytic_auxevol(&commondata, &interp_src_bcstruct);
  if (error_code == BHAHAHA_SUCCESS)
    error_code = counterexample2_ensure_directory_exists(output_dir);
  if (error_code == BHAHAHA_SUCCESS)
    error_code = counterexample2_emit_cartesian_products(&commondata, &griddata[0].params, output_dir);

  if (error_code != BHAHAHA_SUCCESS)
    fprintf(stderr, "Counterexample 2 minimal diagnostic failed: %d (%s)\n", error_code, error_message(error_code));
  else
    printf("Counterexample 2 minimal diagnostic CSVs written to %s\n", output_dir);

  counterexample2_cleanup(&commondata, griddata, &interp_src_bcstruct);
  BHAH_FREE(griddata);
  return error_code;
""",
    )


def register_CFunction_main() -> None:
    """Register the executable entrypoint."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "BHaH_function_prototypes.h"],
        desc="Executable entrypoint for the Counterexample 2 minimal diagnostic.",
        cfunc_type="int",
        name="main",
        params="int argc, const char *argv[]",
        include_CodeParameters_h=False,
        body=r"""
  const char *output_dir = "counterexample2_output";
  if (argc > 1)
    output_dir = argv[1];
  return counterexample2_run(output_dir);
""",
    )


def register_CFunctions() -> None:
    """Register all driver helpers and the main entrypoint."""
    register_CFunction_counterexample2_ensure_directory_exists()
    register_CFunction_counterexample2_cleanup()
    register_CFunction_counterexample2_run()
    register_CFunction_main()
