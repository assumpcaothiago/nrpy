"""
Library of C functions for solving the hyperbolic relaxation equation in curvilinear coordinates, using a reference-metric formalism.

Authors: Thiago Assumpção; assumpcaothiago **at** gmail **dot** com
         Zachariah B. Etienne; zachetie **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from pathlib import Path
from types import FrameType as FT
from typing import Dict, Tuple, Union, cast

import sympy as sp

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.grid as gri
import nrpy.helpers.parallel_codegen as pcg
import nrpy.infrastructures.BHaH.diagnostics.output_0d_1d_2d_nearest_gridpoint_slices as out012d
import nrpy.infrastructures.BHaH.simple_loop as lp
import nrpy.params as par
from nrpy.equations.nrpyelliptic.HeadonNS_RHSs import (
    HyperbolicRelaxationCurvilinearRHSs,
)
from nrpy.equations.nrpyelliptic.HeadonNS_SourceTerms import SourceTerms


# ----------------------------------------------------------------------------------- #
#
#                       Define functions to set up initial guess
#
# ----------------------------------------------------------------------------------- #


def register_CFunction_initial_guess_single_point() -> Union[None, pcg.NRPyEnv_type]:
    """Register the C function for initial guess of solution at a single point.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]

    desc = r"""Compute initial guess at a single point."""
    cfunc_type = "void"
    name = "initial_guess_single_point"
    params = r"""const commondata_struct *restrict commondata, const params_struct *restrict params,
    const REAL xx0, const REAL xx1, const REAL xx2,  REAL *restrict psi_ID, REAL *restrict xi_ID,
    REAL *restrict alphaconf_ID, REAL *restrict tau_ID
"""
    body = ccg.c_codegen(
        4 * [sp.sympify(0)],
        ["*psi_ID", "*xi_ID", "*alphaconf_ID", "*tau_ID"],
        verbose=False,
        include_braces=False,
    )
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=True,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


def register_CFunction_initial_guess_all_points(
    OMP_collapse: int,
    enable_checkpointing: bool = False,
) -> Union[None, pcg.NRPyEnv_type]:
    """Register the initial guess function for the hyperbolic relaxation equation.

    :param enable_checkpointing: Attempt to read from a checkpoint file before generating initial guess.
    :param OMP_collapse: Degree of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None
    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = r"""Set initial guess to solutions of hyperbolic relaxation equation at all points."""
    cfunc_type = "void"
    name = "initial_data"
    params = (
        "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
    )
    psi_gf_memaccess = gri.BHaHGridFunction.access_gf("psi")
    xi_gf_memaccess = gri.BHaHGridFunction.access_gf("xi")
    alphaconf_gf_memaccess = gri.BHaHGridFunction.access_gf("alphaconf")
    tau_gf_memaccess = gri.BHaHGridFunction.access_gf("tau")
    body = ""
    if enable_checkpointing:
        body += """// Attempt to read checkpoint file. If it doesn't exist, then continue. Otherwise return.
if( read_checkpoint(commondata, griddata) ) return;
"""
    body += r"""for(int grid=0; grid<commondata->NUMGRIDS; grid++) {
  // Unpack griddata struct:
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];
  REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
"""
    body += lp.simple_loop(
        loop_body="initial_guess_single_point(commondata, params, xx0,xx1,xx2,"
        f"&{psi_gf_memaccess},"
        f"&{xi_gf_memaccess},"
        f"&{alphaconf_gf_memaccess},"
        f"&{tau_gf_memaccess});",
        read_xxs=True,
        loop_region="all points",
        OMP_collapse=OMP_collapse,
    )
    body += "}\n"
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


# ----------------------------------------------------------------------------------- #
#
#                      Define functions to set AUXEVOL gridfunctions
#
# ----------------------------------------------------------------------------------- #


def register_CFunction_compute_single_source_term(
    SourceType: str = "polytropic_fit",
) -> Union[None, pcg.NRPyEnv_type]:
    """Register the C function to compute single source term.

    :param SourceType: Type of source term. Only valid option: "polytropic_fit".

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]

    desc = r"""Compute single source term as function of radial distance from the center."""
    cfunc_type = "void"
    name = "compute_single_source_term"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL r, REAL *rho, REAL *P"

    source = SourceTerms(SourceType=SourceType)

    body = r"if (r < star_radius)"
    body += ccg.c_codegen(
        [source.rho, source.P],
        ["*rho", "*P"],
        verbose=False,
        include_braces=True,
    )
    body += r"""
  else {
    *rho = 0.0;
    *P = 0.0;
  }"""

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=True,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


def register_CFunction_auxevol_gfs_all_points(
    SourceType: str = "polytropic_fit",
    OMP_collapse: int = 1,
) -> Union[None, pcg.NRPyEnv_type]:
    """Register the C function for the AUXEVOL grid functions at all points.

    :param SourceType: Type of source term.  Only valid option: "polytropic_fit".
    :param OMP_collapse: Degree of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = r"""Set AUXEVOL gridfunctions at all points."""
    cfunc_type = "void"
    name = "auxevol_gfs_all_points"
    params = (
        "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
    )
    rho_memaccess = gri.BHaHGridFunction.access_gf("rho")
    P_memaccess = gri.BHaHGridFunction.access_gf("P")

    body = r"""for(int grid=0; grid<commondata->NUMGRIDS; grid++) {
  // Unpack griddata struct:
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];
  REAL *restrict in_gfs = griddata[grid].gridfuncs.auxevol_gfs;
"""

    loop_body = rf"""
// Compute Cartesian coordinates
REAL xCart[3]; xx_to_Cart(commondata, params, xx, i0, i1, i2, xCart);

// Define x^2 + y^2
const REAL x2_plus_y2 = xCart[0] * xCart[0] + xCart[1] * xCart[1];

// Compute distance from point xCart to each particle at z = z0_pos and z = z1_pos
const REAL r0 = sqrt(x2_plus_y2 + (xCart[2] - z0_pos) * (xCart[2] - z0_pos) );
const REAL r1 = sqrt(x2_plus_y2 + (xCart[2] - z1_pos) * (xCart[2] - z1_pos) );

// Declare density and pressure centered at z0_pos and z1_pos
REAL rho0, rho1, P0, P1;

// Compute individual source terms using the relative positions r0 and r1
compute_single_source_term(commondata, params, r0, &rho0, &P0);
compute_single_source_term(commondata, params, r1, &rho1, &P1);

// Compute superposed source terms
{rho_memaccess} = rho0 + rho1;
{P_memaccess} = P0 + P1;
"""

    body += lp.simple_loop(
        loop_body="\n" + loop_body,
        read_xxs=False,
        loop_region="interior",
        OMP_collapse=OMP_collapse,
    )
    body += "} // END LOOP: for (int grid = 0; grid < commondata->NUMGRIDS; grid++) \n"

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )

    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


# ----------------------------------------------------------------------------------- #
#
#                   Define functions to evaluate RHSs and residuals
#
# ----------------------------------------------------------------------------------- #


def register_CFunction_rhs_eval(
    CoordSystem: str,
    enable_rfm_precompute: bool,
    enable_simd: bool,
    OMP_collapse: int,
) -> Union[None, pcg.NRPyEnv_type]:
    """Register the right-hand side (RHS) evaluation function for the hyperbolic relaxation equation.

    This function sets the right-hand side of the hyperbolic relaxation equation according to the
    selected coordinate system and specified parameters.

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.
    :param enable_simd: Whether to enable SIMD.
    :param OMP_collapse: Level of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    if enable_simd:
        includes += [str(Path("intrinsics") / "simd_intrinsics.h")]
    desc = r"""Set RHSs for hyperbolic relaxation equation."""
    cfunc_type = "void"
    name = "rhs_eval"
    params = "const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs"
    if enable_rfm_precompute:
        params = params.replace(
            "REAL *restrict xx[3]", "const rfm_struct *restrict rfmstruct"
        )
    # Populate uu_rhs, vv_rhs
    rhs = HyperbolicRelaxationCurvilinearRHSs(CoordSystem, enable_rfm_precompute)
    body = lp.simple_loop(
        loop_body=ccg.c_codegen(
            [rhs.psi_rhs, rhs.alphaconf_rhs, rhs.xi_rhs, rhs.tau_rhs],
            [
                gri.BHaHGridFunction.access_gf("psi", gf_array_name="rhs_gfs"),
                gri.BHaHGridFunction.access_gf("alphaconf", gf_array_name="rhs_gfs"),
                gri.BHaHGridFunction.access_gf("xi", gf_array_name="rhs_gfs"),
                gri.BHaHGridFunction.access_gf("tau", gf_array_name="rhs_gfs"),
            ],
            enable_fd_codegen=True,
            enable_simd=enable_simd,
        ),
        loop_region="interior",
        enable_simd=enable_simd,
        CoordSystem=CoordSystem,
        enable_rfm_precompute=enable_rfm_precompute,
        read_xxs=not enable_rfm_precompute,
        OMP_collapse=OMP_collapse,
    )

    cfc.register_CFunction(
        include_CodeParameters_h=True,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        body=body,
        enable_simd=enable_simd,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


def register_CFunction_compute_residual_all_points(
    CoordSystem: str,
    enable_rfm_precompute: bool,
    enable_simd: bool,
    OMP_collapse: int,
) -> Union[None, pcg.NRPyEnv_type]:
    """Register the residual evaluation function.

    This function sets the residual of the Hamiltonian constraint in the hyperbolic
    relaxation equation according to the selected coordinate system and specified
    parameters.

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.
    :param enable_simd: Whether to enable SIMD.
    :param OMP_collapse: Level of OpenMP loop collapsing.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]
    if enable_simd:
        includes += [str(Path("intrinsics") / "simd_intrinsics.h")]
    desc = r"""Compute residual of the Hamiltonian constraint for the hyperbolic relaxation equation."""
    cfunc_type = "void"
    name = "compute_residual_all_points"
    params = """const commondata_struct *restrict commondata, const params_struct *restrict params,
                REAL *restrict xx[3], const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                REAL *restrict aux_gfs"""
    if enable_rfm_precompute:
        params = params.replace(
            "REAL *restrict xx[3]", "const rfm_struct *restrict rfmstruct"
        )
    # Populate residual grid functions
    rhs = HyperbolicRelaxationCurvilinearRHSs(CoordSystem, enable_rfm_precompute)
    body = lp.simple_loop(
        loop_body=ccg.c_codegen(
            [rhs.residual_psi, rhs.residual_alphaconf],
            [
                gri.BHaHGridFunction.access_gf("residual_psi", gf_array_name="aux_gfs"),
                gri.BHaHGridFunction.access_gf(
                    "residual_alphaconf", gf_array_name="aux_gfs"
                ),
            ],
            enable_fd_codegen=True,
            enable_simd=enable_simd,
        ),
        loop_region="interior",
        enable_simd=enable_simd,
        CoordSystem=CoordSystem,
        enable_rfm_precompute=enable_rfm_precompute,
        read_xxs=not enable_rfm_precompute,
        OMP_collapse=OMP_collapse,
    )

    cfc.register_CFunction(
        include_CodeParameters_h=True,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        body=body,
        enable_simd=enable_simd,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


# ----------------------------------------------------------------------------------- #
#
#                               Define diagnostics functions
#
# ----------------------------------------------------------------------------------- #


def register_CFunction_diagnostics(
    CoordSystem: str,
    default_diagnostics_out_every: int,
    enable_progress_indicator: bool = False,
    axis_filename_tuple: Tuple[str, str] = (
        "out1d-AXIS-n-%08d.txt",
        "nn",
    ),
    plane_filename_tuple: Tuple[str, str] = (
        "out2d-PLANE-n-%08d.txt",
        "nn",
    ),
    out_quantities_dict: Union[str, Dict[Tuple[str, str], str]] = "default",
) -> Union[None, pcg.NRPyEnv_type]:
    """Register C function for simulation diagnostics.

    :param CoordSystem: Coordinate system used.
    :param default_diagnostics_out_every: Specifies the default diagnostics output frequency.
    :param enable_progress_indicator: Whether to enable the progress indicator.
    :param axis_filename_tuple: Tuple containing filename and variables for axis output.
    :param plane_filename_tuple: Tuple containing filename and variables for plane output.
    :param out_quantities_dict: Dictionary or string specifying output quantities.
    :return: None if in registration phase, else the updated NRPy environment.
    :raises TypeError: If `out_quantities_dict` is not a dictionary and not set to "default".
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    _ = par.CodeParameter(
        "int",
        __name__,
        "diagnostics_output_every",
        default_diagnostics_out_every,
        commondata=True,
    )

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]
    desc = "Diagnostics."
    cfunc_type = "void"
    name = "diagnostics"
    params = (
        "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
    )

    # Memory access for evolution gridfunctions
    psi_gf_memaccess = gri.BHaHGridFunction.access_gf("psi", gf_array_name="y_n_gfs")
    alphaconf_gf_memaccess = gri.BHaHGridFunction.access_gf(
        "alphaconf", gf_array_name="y_n_gfs"
    )

    # Memory access for residuals
    residual_psi_gf_memaccess = gri.BHaHGridFunction.access_gf(
        "residual_psi", gf_array_name="diagnostic_output_gfs"
    )
    residual_alphaconf_gf_memaccess = gri.BHaHGridFunction.access_gf(
        "residual_alphaconf", gf_array_name="diagnostic_output_gfs"
    )

    # Memory access for source terms
    rho_gf_memaccess = gri.BHaHGridFunction.access_gf(
        "rho", gf_array_name="auxevol_gfs"
    )
    P_gf_memaccess = gri.BHaHGridFunction.access_gf("P", gf_array_name="auxevol_gfs")

    # fmt: off
    if out_quantities_dict == "default":
        out_quantities_dict = {
            ("REAL", "num_psi"): psi_gf_memaccess,
            ("REAL", "num_alphaconf"): alphaconf_gf_memaccess,
            ("REAL", "num_residual_psi"): residual_psi_gf_memaccess,
            ("REAL", "num_residual_alphaconf"): residual_alphaconf_gf_memaccess,
            ("REAL", "num_rho"): rho_gf_memaccess,
            ("REAL", "num_P"): P_gf_memaccess,
        }
    if not isinstance(out_quantities_dict, dict):
        raise TypeError(f"out_quantities_dict was initialized to {out_quantities_dict}, which is not a dictionary!")
    # fmt: on

    for axis in ["y", "z"]:
        out012d.register_CFunction_diagnostics_nearest_1d_axis(
            CoordSystem=CoordSystem,
            out_quantities_dict=out_quantities_dict,
            filename_tuple=axis_filename_tuple,
            axis=axis,
        )
    for plane in ["xy", "yz"]:
        out012d.register_CFunction_diagnostics_nearest_2d_plane(
            CoordSystem=CoordSystem,
            out_quantities_dict=out_quantities_dict,
            filename_tuple=plane_filename_tuple,
            plane=plane,
        )

    body = r"""  // Output progress to stderr
  progress_indicator(commondata, griddata);

  // Since this version of NRPyElliptic is unigrid, we simply set the grid index to 0
  const int grid = 0;

  // Set gridfunctions aliases
  REAL *restrict y_n_gfs = griddata[grid].gridfuncs.y_n_gfs;
  REAL *restrict auxevol_gfs = griddata[grid].gridfuncs.auxevol_gfs;
  REAL *restrict diagnostic_output_gfs = griddata[grid].gridfuncs.diagnostic_output_gfs;

  // Set params and rfm_struct
  params_struct *restrict params = &griddata[grid].params;
  const rfm_struct *restrict rfmstruct = griddata[grid].rfmstruct;
#include "set_CodeParameters.h"

  // Compute residuals and store them at diagnostic_output_gfs
  compute_residual_all_points(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, diagnostic_output_gfs);

  // Set integration radius for l2-norm computation
  const REAL integration_radius = 1000.0;

  // Compute l2-norm of both residuals -- psi and alphaconf
  const REAL residual_psi = compute_L2_norm_of_gridfunction(commondata, griddata, integration_radius,
                                                            RESIDUAL_PSIGF, diagnostic_output_gfs);
  const REAL residual_alphaconf = compute_L2_norm_of_gridfunction(commondata, griddata, integration_radius,
                                                                  RESIDUAL_ALPHACONFGF, diagnostic_output_gfs);

  const REAL total_residual = residual_psi + residual_alphaconf;

  // Update residual to be used in stop condition
  commondata->log10_current_residual = total_residual;

  // Output l2-norm of sum of residuals constraint violation to file
  {
    char filename[256];
    sprintf(filename, "residual_l2_norm.txt");
    FILE *outfile = (nn == 0) ? fopen(filename, "w") : fopen(filename, "a");
    if (!outfile) {
      fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
      exit(1);
    }
    fprintf(outfile, "%6d %10.4e %.17e\n", nn, time, total_residual);
    fclose(outfile);
  }

  // Grid data output
  const int n_step = commondata->nn, outevery = commondata->diagnostics_output_every;
  if (n_step % outevery == 0) {
    // Set reference metric grid xx
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
        xx[ww] = griddata[grid].xx[ww];

    // 1D output
    diagnostics_nearest_1d_y_axis(commondata, params, xx, &griddata[grid].gridfuncs);
    diagnostics_nearest_1d_z_axis(commondata, params, xx, &griddata[grid].gridfuncs);

    // 2D output
    diagnostics_nearest_2d_xy_plane(commondata, params, xx, &griddata[grid].gridfuncs);
    diagnostics_nearest_2d_yz_plane(commondata, params, xx, &griddata[grid].gridfuncs);
  }
"""
    if enable_progress_indicator:
        body += "progress_indicator(commondata, griddata);"
    body += r"""
  if (commondata->time + commondata->dt > commondata->t_final)
    printf("\n");
"""

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


# if __name__ == "__main__":

#     register_CFunction_compute_single_source_term()
#     # Print function
#     print(cfc.CFunction_dict["compute_single_source_term"].full_function)

# register_CFunction_auxevol_gfs_all_points()
# # Print function
# print(cfc.CFunction_dict["auxevol_gfs_all_points"].full_function)

# register_CFunction_rhs_eval(
#     CoordSystem="SinhSymTP",
#     enable_rfm_precompute=False,
#     enable_simd=False,
#     OMP_collapse=1,
# )
# # Print function
# print(cfc.CFunction_dict["rhs_eval"].full_function)

# register_CFunction_compute_residual_all_points(
#     CoordSystem="SinhSymTP",
#     enable_rfm_precompute=False,
#     enable_simd=False,
#     OMP_collapse=1,
# )
# # Print function
# print(cfc.CFunction_dict["compute_residual_all_points"].full_function)

# register_CFunction_diagnostics(
#     CoordSystem="SinhSymTP",
#     default_diagnostics_out_every=1,
#     enable_progress_indicator=True,
# )
# # Print function
# print(cfc.CFunction_dict["diagnostics"].full_function)
