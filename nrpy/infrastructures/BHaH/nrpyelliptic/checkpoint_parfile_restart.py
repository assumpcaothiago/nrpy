"""
Register permissive parfile reparse support after NRPyElliptic checkpoint reads.

The generated C helper is intended for restart workflows where the checkpoint
owns the solution state, but the user may still want to steer runtime/output
parameters from the parfile before either continuing relaxation or writing
post-processed interpolation output.
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import List, Tuple, Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg

Param = Tuple[str, str]


def _field_expr(struct_name: str, field: str) -> str:
    return f"{struct_name}.{field}"


def _generate_disallowed_checks(params: List[Param]) -> str:
    checks = []
    for name, param_type in params:
        old = _field_expr("checkpoint_commondata", name)
        new = _field_expr("parfile_commondata", name)
        if param_type == "REAL":
            check = (
                f"  if (checkpoint_REAL_changed({old}, {new})) {{\n"
                f'    checkpoint_report_disallowed_REAL_update("{name}", {old}, {new});\n'
                "    disallowed_changes++;\n"
                "  }\n"
            )
        elif param_type == "char":
            check = (
                f"  if (checkpoint_string_changed({old}, {new})) {{\n"
                f'    checkpoint_report_disallowed_string_update("{name}", {old}, {new});\n'
                "    disallowed_changes++;\n"
                "  }\n"
            )
        else:
            raise ValueError(f"Unsupported disallowed parameter type: {param_type}")
        checks.append(check)
    return "".join(checks)


def _generate_allowed_reports(params: List[Param]) -> str:
    reports = []
    for name, param_type in params:
        old = _field_expr("checkpoint_commondata", name)
        new = _field_expr("parfile_commondata", name)
        if param_type == "REAL":
            report = (
                f"  if (checkpoint_REAL_changed({old}, {new})) {{\n"
                f'    checkpoint_report_REAL_update("{name}", {old}, {new});\n'
                "  }\n"
            )
        elif param_type == "int":
            if name == "checkpoint_every":
                report = (
                    f"  if ({old} != {new} && {old} != 1) {{\n"
                    f'    checkpoint_report_int_update("{name}", {old}, {new});\n'
                    "  }\n"
                )
            else:
                report = (
                    f"  if ({old} != {new}) {{\n"
                    f'    checkpoint_report_int_update("{name}", {old}, {new});\n'
                    "  }\n"
                )
        elif param_type == "bool":
            report = (
                f"  if ({old} != {new}) {{\n"
                f'    checkpoint_report_bool_update("{name}", {old}, {new});\n'
                "  }\n"
            )
        elif param_type == "char":
            report = (
                f"  if (checkpoint_string_changed({old}, {new})) {{\n"
                f'    checkpoint_report_string_update("{name}", {old}, {new});\n'
                "  }\n"
            )
        else:
            raise ValueError(f"Unsupported allowed parameter type: {param_type}")
        reports.append(report)
    return "".join(reports)


def register_CFunction_checkpoint_apply_parfile_updates_after_restart() -> (
    Union[None, pcg.NRPyEnv_type]
):
    """
    Register a C helper that reapplies compatible parfile values after restart.

    The function:
      - returns immediately when no checkpoint was read;
      - parses the active parfile/CLI arguments into a temporary commondata;
      - errors out if source/grid/boundary compatibility parameters changed;
      - reports and applies allowed runtime/output/interpolation changes;
      - preserves checkpoint-owned solution state;
      - recomputes dt and stop_relaxation;
      - writes interpolation immediately when the restarted state already stops.

    :return: None during parallel-codegen registration, else updated NRPyEnv.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    disallowed_params: List[Param] = [
        ("P0_x", "REAL"),
        ("P0_y", "REAL"),
        ("P0_z", "REAL"),
        ("P1_x", "REAL"),
        ("P1_y", "REAL"),
        ("P1_z", "REAL"),
        ("S0_x", "REAL"),
        ("S0_y", "REAL"),
        ("S0_z", "REAL"),
        ("S1_x", "REAL"),
        ("S1_y", "REAL"),
        ("S1_z", "REAL"),
        ("bare_mass_0", "REAL"),
        ("bare_mass_1", "REAL"),
        ("xPunc", "REAL"),
        ("zPunc", "REAL"),
        ("convergence_factor", "REAL"),
        ("outer_bc_type", "char"),
    ]
    allowed_params: List[Param] = [
        ("CFL_FACTOR", "REAL"),
        ("MINIMUM_GLOBAL_WAVESPEED", "REAL"),
        ("eta_damping", "REAL"),
        ("t_final", "REAL"),
        ("checkpoint_every", "int"),
        ("compute_residual_every", "int"),
        ("diagnostics_nearest_output_every", "int"),
        ("output_progress_every", "int"),
        ("axis_interpolation_enable", "bool"),
        ("axis_interpolation_axis", "char"),
        ("axis_interpolation_num_points", "int"),
        ("axis_interpolation_x_min", "REAL"),
        ("axis_interpolation_x_max", "REAL"),
        ("axis_interpolation_x_offset", "REAL"),
        ("axis_interpolation_y_offset", "REAL"),
        ("axis_interpolation_z_min", "REAL"),
        ("axis_interpolation_z_max", "REAL"),
        ("axis_interpolation_z_offset", "REAL"),
        ("nn_max", "int"),
        ("log10_residual_tolerance", "REAL"),
    ]

    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
        "math.h",
        "stdio.h",
        "stdlib.h",
        "string.h",
    ]
    desc = """
Reparse compatible parfile values after a checkpoint restart.

The checkpoint owns the numerical solution and runtime state. The parfile may
override runtime, stopping, output, and interpolation settings. The function
rejects changes that would make the checkpoint inconsistent with the equations,
source terms, grid layout, or boundary setup.
"""
    cfunc_type = "void"
    name = "checkpoint_apply_parfile_updates_after_restart"
    params = (
        "commondata_struct *restrict commondata, const int argc, "
        "const char *argv[], const int checkpoint_has_been_read, "
        "griddata_struct *restrict griddata"
    )
    prefunc = r"""
static int checkpoint_REAL_changed(const REAL old_value, const REAL new_value) {
  return old_value != new_value;
}

static int checkpoint_string_changed(const char *old_value, const char *new_value) {
  return strcmp(old_value, new_value) != 0;
}

static const char *checkpoint_bool_string(const bool value) {
  return value ? "True" : "False";
}

static void checkpoint_report_REAL_update(const char *name, const REAL old_value, const REAL new_value) {
  printf("checkpoint parfile update: %s: %.17g -> %.17g\n", name, (double)old_value, (double)new_value);
  fflush(stdout);
}

static void checkpoint_report_int_update(const char *name, const int old_value, const int new_value) {
  printf("checkpoint parfile update: %s: %d -> %d\n", name, old_value, new_value);
  fflush(stdout);
}

static void checkpoint_report_bool_update(const char *name, const bool old_value, const bool new_value) {
  printf("checkpoint parfile update: %s: %s -> %s\n", name, checkpoint_bool_string(old_value), checkpoint_bool_string(new_value));
  fflush(stdout);
}

static void checkpoint_report_string_update(const char *name, const char *old_value, const char *new_value) {
  printf("checkpoint parfile update: %s: \"%s\" -> \"%s\"\n", name, old_value, new_value);
  fflush(stdout);
}

static void checkpoint_report_disallowed_REAL_update(const char *name, const REAL old_value, const REAL new_value) {
  fprintf(stderr, "checkpoint parfile error: disallowed change to %s: %.17g -> %.17g\n", name, (double)old_value, (double)new_value);
}

static void checkpoint_report_disallowed_string_update(const char *name, const char *old_value, const char *new_value) {
  fprintf(stderr, "checkpoint parfile error: disallowed change to %s: \"%s\" -> \"%s\"\n", name, old_value, new_value);
}

"""

    body = f"""
  if (!checkpoint_has_been_read) {{
    return;
  }}

  const commondata_struct checkpoint_commondata = *commondata;

  commondata_struct parfile_commondata;
  commondata_struct_set_to_default(&parfile_commondata);
  cmdline_input_and_parfile_parser(&parfile_commondata, argc, argv);

  int disallowed_changes = 0;
{_generate_disallowed_checks(disallowed_params)}
  if (disallowed_changes > 0) {{
    fprintf(stderr,
            "checkpoint parfile error: refusing to reuse checkpoint with %d incompatible parameter change%s.\\n",
            disallowed_changes, disallowed_changes == 1 ? "" : "s");
    exit(1);
  }}

{_generate_allowed_reports(allowed_params)}
  *commondata = parfile_commondata;

  // Preserve checkpoint-owned state. These fields describe the restored
  // solution, not the user's new runtime steering choices.
  commondata->NUMGRIDS = checkpoint_commondata.NUMGRIDS;
  commondata->dt = checkpoint_commondata.dt;
  commondata->t_0 = checkpoint_commondata.t_0;
  commondata->time = checkpoint_commondata.time;
  commondata->nn = checkpoint_commondata.nn;
  commondata->nn_0 = checkpoint_commondata.nn_0;
  commondata->log10_current_residual = checkpoint_commondata.log10_current_residual;
  commondata->start_wallclock_time = checkpoint_commondata.start_wallclock_time;

  // Recompute dt so an allowed CFL_FACTOR update is honored for any continued
  // relaxation from this checkpoint.
  commondata->dt = 1e30;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {{
    cfl_limited_timestep(commondata, &griddata[grid].params, griddata[grid].xx);
  }}

  commondata->stop_relaxation = false;
  stop_conditions_check(commondata);
  if (commondata->stop_relaxation) {{
    printf("checkpoint parfile update: stop condition already satisfied after checkpoint restore; writing interpolated output.\\n");
    fflush(stdout);
    axis_interpolation_1d_output(commondata, griddata);
    commondata->t_final = commondata->time;
  }}
"""
    cfc.register_CFunction(
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return pcg.NRPyEnv()


if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
