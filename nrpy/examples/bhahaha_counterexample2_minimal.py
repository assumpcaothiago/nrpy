"""
Generate a minimal NRPy-only Counterexample 2 diagnostic that uses standard
BHaH griddata storage and the current unfixed BHaHAHA derivative path.
"""

import argparse
import os
import pkgutil
import shutil
from pathlib import Path

import nrpy.c_function as cfc
import nrpy.params as par
import nrpy.reference_metric as refmetric
from nrpy.infrastructures import BHaH
from nrpy.infrastructures import derivative_interpolation_test as dit


parser = argparse.ArgumentParser(
    description="Generate a minimal Counterexample 2 BHaHAHA diagnostic project."
)
parser.add_argument(
    "--fdorder",
    type=int,
    default=6,
    help="Finite-difference order used for derivative stencils. Default=6.",
)
parser.add_argument(
    "--outrootdir",
    type=str,
    default="project",
    help="Output root directory. Default=project/",
)
parser.add_argument(
    "--project-name",
    type=str,
    default="BHaHAHA-counterexample2-minimal",
    help="Generated project directory name.",
)
parser.add_argument(
    "--nr",
    type=int,
    default=64,
    help="Number of positive-r source-grid interior points. Default=64.",
)
parser.add_argument(
    "--ntheta",
    type=int,
    default=65,
    help="Number of theta source-grid interior points. Must be odd so theta=pi/2 is cell-centered. Default=65.",
)
parser.add_argument(
    "--nphi",
    type=int,
    default=64,
    help="Number of phi source-grid interior points. Must be even so phi -> phi+pi maps to cell centers across theta-axis BCs. Default=64.",
)
parser.add_argument(
    "--rmax",
    type=float,
    default=4.5,
    help="Positive radial extent of the source grid. Default=4.5.",
)
parser.add_argument(
    "--cartesian-half-width",
    type=float,
    default=3.0,
    help="Half-width of the Cartesian lines and planes. Default=3.0.",
)
parser.add_argument(
    "--no-openmp",
    action="store_true",
    help="Disable OpenMP in the generated Makefile.",
)
args = parser.parse_args()

if args.ntheta % 2 == 0:
    raise ValueError("--ntheta must be odd so theta=pi/2 is a cell center.")
if args.nphi % 2 != 0:
    raise ValueError(
        "--nphi must be even so the current parity-based theta-axis BCs map phi -> phi+pi onto cell centers."
    )

CoordSystem = "Spherical"
project_dir = os.path.join(args.outrootdir, args.project_name)

shutil.rmtree(project_dir, ignore_errors=True)

par.set_parval_from_str("Infrastructure", "BHaH")
par.set_parval_from_str("parallelization", "openmp")
par.set_parval_from_str("fp_type", "double")
par.set_parval_from_str("enable_parallel_codegen", False)
par.set_parval_from_str("fd_order", args.fdorder)
par.set_parval_from_str("CoordSystem_to_register_CodeParameters", CoordSystem)

_ = refmetric.reference_metric[CoordSystem]

BHaH.BHaH_defines_h.register_BHaH_defines("after_general", '#include "BHaHAHA.h"\n')
dit.defines.register_error_codes()
dit.defines.register_BHaH_defines(args.cartesian_half_width)
dit.gridfunctions.register_gridfunctions()

BHaH.BHaHAHA.error_message.register_CFunction_error_message()
BHaH.BHaHAHA.interpolation_3d_general__uniform_src_grid.register_CFunction_interpolation_3d_general__uniform_src_grid(
    enable_simd=False,
    project_dir=project_dir,
)
BHaH.BHaHAHA.hDD_dD_and_W_dD_in_interp_src_grid_interior.register_CFunction_hDD_dD_and_W_dD_in_interp_src_grid_interior()
BHaH.BHaHAHA.apply_bcs_r_maxmin_partial_r_hDD_upwinding.register_CFunction_apply_bcs_r_maxmin_partial_r_hDD_upwinding(
    upwinding_fd_order=args.fdorder
)
BHaH.BHaHAHA.bcstruct_set_up.register_CFunction_bcstruct_set_up(CoordSystem=CoordSystem)
BHaH.BHaHAHA.numgrid__interp_src_set_up.register_CFunction_numgrid__interp_src_set_up()
cfc.CFunction_dict.pop("numgrid__interp_src_set_up", None)

BHaH.numerical_grids_and_timestep.register_CFunctions(
    set_of_CoordSystems={CoordSystem},
    list_of_grid_physical_sizes=[args.rmax],
    Nxx_dict={CoordSystem: [args.nr, args.ntheta, args.nphi]},
    gridding_approach="independent grid(s)",
    enable_rfm_precompute=False,
    enable_CurviBCs=False,
    enable_set_cfl_timestep=False,
)
BHaH.xx_tofrom_Cart.register_CFunction__Cart_to_xx_and_nearest_i0i1i2(
    CoordSystem=CoordSystem
)
BHaH.rfm_wrapper_functions.register_CFunctions_CoordSystem_wrapper_funcs()

dit.compatibility_aliases.register_BHaH_defines()

dit.allocation.register_CFunction_counterexample2_allocate_auxevol_gfs()
dit.interp_src_binding.register_CFunction_counterexample2_bind_griddata_to_interp_src()
dit.analytic_pipeline.register_CFunctions()
dit.reconstruct_dhdx.register_CFunction_counterexample2_compute_dhdx()
dit.cartesian_products.register_CFunctions()
dit.driver.register_CFunctions()

BHaH.CodeParameters.write_CodeParameters_h_files(project_dir=project_dir)
BHaH.CodeParameters.register_CFunctions_params_commondata_struct_set_to_default()
BHaH.BHaH_defines_h.output_BHaH_defines_h(
    project_dir=project_dir,
    enable_rfm_precompute=False,
)

BHaH.Makefile_helpers.output_CFunctions_function_prototypes_and_construct_Makefile(
    project_dir=project_dir,
    project_name=args.project_name,
    exec_or_library_name=args.project_name,
    use_openmp=not args.no_openmp,
)

data_bytes = pkgutil.get_data("nrpy.infrastructures.BHaH.BHaHAHA", "BHaHAHA_header.h")
if data_bytes is None:
    raise FileNotFoundError("BHaHAHA_header.h not found via pkgutil.get_data")

BHaHAHA_h = data_bytes.decode("utf-8")
BHaHAHA_h += f"""
//===============================================
// Set the number of (finite-difference) ghostzones in BHaHAHA
//===============================================
#define BHAHAHA_NGHOSTS {int(par.parval_from_str("finite_difference::fd_order") / 2)}
"""
BHaHAHA_h += """
//===============================================
// BHaHAHA error handling
//===============================================
typedef enum {
"""
for item in BHaH.BHaHAHA.error_message.error_code_msg_tuples_list:
    BHaHAHA_h += f"  {item[0]},\n"
BHaHAHA_h += "} bhahaha_error_codes;\n"
BHaHAHA_h += """
const char *error_message(const bhahaha_error_codes error_code);
#define bah_error_message error_message
//===============================================

#endif // BHAHAHA_HEADER_H
"""

with Path(project_dir, "BHaHAHA.h").open("w", encoding="utf-8") as output_file:
    output_file.write(BHaHAHA_h)

print(
    f"Finished! Now go into ./{args.outrootdir}/{args.project_name} and type `make` to build the diagnostic,"
    f" then ./{args.project_name} [output_dir] to run it."
)
