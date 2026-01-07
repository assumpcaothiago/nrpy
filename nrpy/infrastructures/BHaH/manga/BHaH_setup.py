# nrpy/infrastructures/BHaH/manga/BHaH_setup.py
"""
Generate function to allocate memory and intialize BHaH quantities

Author: Thiago Assumpção
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
import nrpy.params as par


def _generate_BHaH_setup_body(
    set_initial_data_after_auxevol_malloc: bool,
    post_non_y_n_auxevol_mallocs: str,
    sinh_params: list[str],
) -> str:
    """
    Generate the C code for the body of the BHaH_setup() function.

    This function constructs the BHaH_setup C function body in a similar
    way that is done in nrpy/infrastructures/BHaH/main_c.py.

    :param set_initial_data_after_auxevol_malloc: Flag to set initial data after auxevol malloc.
    :param post_non_y_n_auxevol_mallocs: String of post-malloc function calls.
    :param sinh_params: List of strings with parameters in the reference-metric coordinates.
    :return: The formatted C code for the function body.
    """
    # Prepare frequently used strings and logic based on parallelization settings.
    parallelization = par.parval_from_str("parallelization")
    is_cuda = parallelization == "cuda"
    compute_griddata = "griddata_device" if is_cuda else "griddata"

    # The body of the main() function is built as a list of C code strings.
    body_parts = []

    # Step 0: Variable declarations
    declarations_c_code = "commondata_struct commondata; // commondata contains parameters common to all grids.\n"
    declarations_c_code += f"griddata_struct *{compute_griddata}; // griddata contains data specific to an individual grid.\n"
    if is_cuda:
        declarations_c_code += r"""griddata_struct *griddata_host; // stores only the host data needed for diagnostics
#include "BHaH_CUDA_global_init.h"
"""
    body_parts.append(declarations_c_code)

    # Step 1: Initialization
    griddata_malloc_code = f"{compute_griddata} = (griddata_struct *)malloc(sizeof(griddata_struct) * MAXNUMGRIDS);"
    if is_cuda:
        griddata_malloc_code += "\ngriddata_host = (griddata_struct *)malloc(sizeof(griddata_struct) * MAXNUMGRIDS);"
    numerical_grids_args = (
        f"&commondata, {compute_griddata}"
        + (", griddata_host" if is_cuda else "")
        + ", calling_for_first_time"
    )

    step1_c_code = f"""
// Step 1.a: Initialize each CodeParameter in the commondata struct to its default value.
commondata_struct_set_to_default(&commondata);

// Step 1.b: Overwrite the default values with those from the parameter file.
//           Then overwrite the parameter file values with those provided via command line arguments.
// cmdline_input_and_parfile_parser(&commondata, argc, argv); -> Thiago says: ignore it for library

// Step 1.c: Allocate memory for MAXNUMGRIDS griddata structs,
//           where each structure contains data specific to an individual grid.
{griddata_malloc_code}

// Step 1.d: Initialize each CodeParameter in {compute_griddata}.params to its default value.
params_struct_set_to_default(&commondata, {compute_griddata});
"""
    step1_c_code += f"""
// Step 1.e: Overwrite grid parameters and CFL factor using values passed by the function call
{{
const int grid = 0;

commondata.CFL_FACTOR = cfl;
(griddata[grid].params).Nxx0 = nxx0;
(griddata[grid].params).Nxx1 = nxx1;
(griddata[grid].params).Nxx2 = nxx2;
(griddata[grid].params).grid_physical_size = grid_physical_size;
"""
    for param in sinh_params:
        step1_c_code += f"(griddata[grid].params).{param} = {param};"

    step1_c_code += f"""
}}

// Step 1.f: Set up numerical grids, including parameters such as NUMGRIDS, xx[3], masks, Nxx, dxx, invdxx,
//           bcstruct, rfm_precompute, timestep, and others.
{{
  IFCUDARUN(for (int grid = 0; grid < MAXNUMGRIDS; grid++) griddata_device[grid].params.is_host = false;);
  // If this function is being called for the first time, initialize commondata.time, nn, t_0, and nn_0 to 0.
  const bool calling_for_first_time = true;
  numerical_grids_and_timestep({numerical_grids_args});
}} // END setup of numerical & temporal grids.
"""
    body_parts.append(step1_c_code)

    allocator_macro = (
        "BHAH_MALLOC_DEVICE" if parallelization == "cuda" else "BHAH_MALLOC"
    )
    # Step 2: Allocate memory for evolved gridfunctions (y_n_gfs).
    step2_c_code = f"""
// Step 2: Allocate storage for the initial data (y_n_gfs gridfunctions) on each grid.
for(int grid=0; grid<commondata.NUMGRIDS; grid++) {{
  const int Nxx_plus_2NGHOSTS_tot = ({compute_griddata}[grid].params.Nxx_plus_2NGHOSTS0 * //
                                     {compute_griddata}[grid].params.Nxx_plus_2NGHOSTS1 * //
                                     {compute_griddata}[grid].params.Nxx_plus_2NGHOSTS2);
  {allocator_macro}({compute_griddata}[grid].gridfuncs.y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_EVOL_GFS);
  if (NUM_AUXEVOL_GFS > 0) {{
    {allocator_macro}({compute_griddata}[grid].gridfuncs.auxevol_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_AUXEVOL_GFS);
    IFCUDARUN(BHAH_MALLOC(griddata_host[grid].gridfuncs.auxevol_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_AUXEVOL_GFS););
  }} // END IF NUM_AUXEVOL_GFS > 0

  // On GPU, separately allocate y_n_gfs on the host, for diagnostics purposes.
  IFCUDARUN(BHAH_MALLOC_PINNED(griddata_host[grid].gridfuncs.y_n_gfs, sizeof(REAL) * Nxx_plus_2NGHOSTS_tot * NUM_EVOL_GFS););
}} // END LOOP over grids
"""
    body_parts.append(step2_c_code)

    # Steps 3 & 4: Set initial data and allocate memory for auxiliary gridfunctions.
    initial_data_call = f"initial_data(&commondata, {f'griddata_host, {compute_griddata}' if is_cuda else compute_griddata});"
    setup_initial_data_code = f"Set up initial data.\n{initial_data_call}\n"
    allocate_storage_code = f"""Allocate storage for non-y_n gridfunctions, needed for the Runge-Kutta-like timestepping.
for(int grid=0; grid<commondata.NUMGRIDS; grid++)
  MoL_malloc_intermediate_stage_gfs(&commondata, &{compute_griddata}[grid].params, &{compute_griddata}[grid].gridfuncs);\n"""

    if set_initial_data_after_auxevol_malloc:
        step3_code, step4_code = (allocate_storage_code, setup_initial_data_code)
    else:
        step3_code, step4_code = (setup_initial_data_code, allocate_storage_code)
    body_parts.append(f"\n// Step 3: {step3_code}\n// Step 4: {step4_code}")

    if post_non_y_n_auxevol_mallocs:
        body_parts.append(
            "// Step 4.a: Functions called after memory for non-y_n and auxevol gridfunctions is allocated.\n"
        )
        body_parts.append(post_non_y_n_auxevol_mallocs)

    step5_c_code = f"""
  // Step 5: Set pointers to griddata and commondata
  bhahstruct->griddata = griddata;
  bhahstruct->commondata = &commondata;
"""
    body_parts.append(step5_c_code)

    #     # Step 6: Free all allocated memory
    #     device_sync = "BHAH_DEVICE_SYNC();" if is_cuda else ""
    #     if not is_cuda:
    #         free_memory_code = rf"""
    #   const bool free_non_y_n_gfs_and_core_griddata_pointers=true;
    #   griddata_free(&commondata, {compute_griddata}, free_non_y_n_gfs_and_core_griddata_pointers);
    # }}
    #         """
    #     else:
    #         free_memory_code = rf"""
    #   const bool free_non_y_n_gfs_and_core_griddata_pointers=true;
    #   griddata_free_device(&commondata, {compute_griddata}, free_non_y_n_gfs_and_core_griddata_pointers);
    #   griddata_free(&commondata, griddata_host, free_non_y_n_gfs_and_core_griddata_pointers);
    # }}
    # for (int i = 0; i < NUM_STREAMS; ++i) {{
    #   cudaStreamDestroy(streams[i]);
    # }}
    # BHAH_DEVICE_SYNC();
    # cudaDeviceReset();
    # """
    #     body_parts.append(
    #         f"""
    # }} // End main loop to progress forward in time.
    # {device_sync}
    # // Step 6: Free all allocated memory
    # {{{free_memory_code}"""
    #     )
    #     body_parts.append(
    #         r"""return 0;
    # """
    #     )

    # Construct the final body string and perform necessary replacements.
    body = "".join(body_parts)
    return body


def register_CFunction_BHaH_setup(
    set_initial_data_after_auxevol_malloc: bool,
    post_non_y_n_auxevol_mallocs: str,
    CoordSystem: str,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to compute rescaled 3-vector in curvilinar coordinates from 3-vector in Cartesian coordinates

    :param CoordSystem: The coordinate system.
    :param enable_rfm_precompute: Whether to enable reference metric precomputation.

    :return: None if in registration phase, else the updated NRPy environment.

    ::note: have to turn off simd, since this is a point wise operation
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "Allocate memory and intialize BHaH quantities."

    cfunc_type = "void"
    name = "BHaH_setup"

    sinh_params=[]
    for key in par.glb_code_params_dict.keys():
        if "sinh" in key.lower():
            sinh_params.append(key)
    sinh_params.sort()
    param_str = ""
    for param in sinh_params:
        param_str += f"const REAL {param}, "


    params = fr"""
        const int nxx0, const int nxx1, const int nxx2, const REAL cfl,
        const REAL grid_physical_size, {param_str}BHaH_struct *bhahstruct
"""

    body = _generate_BHaH_setup_body(
        set_initial_data_after_auxevol_malloc=set_initial_data_after_auxevol_malloc,
        post_non_y_n_auxevol_mallocs=post_non_y_n_auxevol_mallocs,
        sinh_params=sinh_params,
    )


    cfc.register_CFunction(
        include_CodeParameters_h=False,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
    )
    return pcg.NRPyEnv()
