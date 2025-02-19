"""
Library of commonly used C functions for solving the hyperbolic relaxation equation.

Authors: Thiago Assumpção
         assumpcaothiago **at** gmail **dot** com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import sympy as sp

import nrpy.c_codegen as ccg
import nrpy.c_function as cfc
import nrpy.grid as gri
import nrpy.helpers.parallel_codegen as pcg
import nrpy.infrastructures.BHaH.simple_loop as lp
import nrpy.params as par
import nrpy.reference_metric as refmetric


def register_CFunction_variable_wavespeed_gfs_all_points(
    CoordSystem: str,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to compute variable wavespeed based on local grid spacing for a single coordinate system.

    :param CoordSystem: The coordinate system to use in the hyperbolic relaxation.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None
    includes = ["BHaH_defines.h"]
    desc = "Compute variable wavespeed for all grids based on local grid spacing."
    cfunc_type = "void"
    name = "variable_wavespeed_gfs_all_points"
    params = (
        "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
    )

    rfm = refmetric.reference_metric[CoordSystem]
    dxx0, dxx1, dxx2 = sp.symbols("dxx0 dxx1 dxx2", real=True)
    dsmin_computation_str = ccg.c_codegen(
        [
            rfm.scalefactor_orthog[0] * dxx0,
            rfm.scalefactor_orthog[1] * dxx1,
            rfm.scalefactor_orthog[2] * dxx2,
        ],
        ["const REAL dsmin0", "const REAL dsmin1", "const REAL dsmin2"],
        include_braces=False,
    )

    variable_wavespeed_memaccess = gri.BHaHGridFunction.access_gf("variable_wavespeed")

    dsmin_computation_str += f"""\n// Set local wavespeed
        {variable_wavespeed_memaccess} = MINIMUM_GLOBAL_WAVESPEED * MIN(dsmin0, MIN(dsmin1, dsmin2)) / dt;\n"""

    body = r"""for(int grid=0; grid<commondata->NUMGRIDS; grid++) {
  // Unpack griddata struct:
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];
  REAL *restrict in_gfs = griddata[grid].gridfuncs.auxevol_gfs;
"""

    body += lp.simple_loop(
        loop_body="\n" + dsmin_computation_str,
        read_xxs=True,
        loop_region="interior",
    )

    # We must close the loop that was opened in the line 'for(int grid=0; grid<commondata->NUMGRIDS; grid++) {'
    body += r"""} // END LOOP for(int grid=0; grid<commondata->NUMGRIDS; grid++)
            """

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,  # set_CodeParameters.h is manually included after the declaration of params_struct *restrict params
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


def register_CFunction_initialize_constant_auxevol() -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to call all functions that set up AUXEVOL gridfunctions.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = r"""Call functions that set up all AUXEVOL gridfunctions."""
    cfunc_type = "void"
    name = "initialize_constant_auxevol"
    params = (
        "commondata_struct *restrict commondata, griddata_struct *restrict griddata"
    )

    body = r"""
    // Set up variable wavespeed
    variable_wavespeed_gfs_all_points(commondata, griddata);

    // Set up all other AUXEVOL gridfunctions
    auxevol_gfs_all_points(commondata, griddata);
    """

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())


def register_CFunction_compute_L2_norm_of_gridfunction(
    CoordSystem: str,
) -> None:
    """
    Register function to compute l2-norm of a gridfunction assuming a single grid.

    Note that parallel codegen is disabled for this function, as it sometimes causes a
    multiprocess race condition on Python 3.6.7

    :param CoordSystem: the rfm coordinate system.
    """
    includes = ["BHaH_defines.h"]
    desc = "Compute l2-norm of a gridfunction assuming a single grid."
    cfunc_type = "REAL"
    name = "compute_L2_norm_of_gridfunction"
    params = """commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                const REAL integration_radius, const int gf_index, const REAL *restrict in_gf"""

    rfm = refmetric.reference_metric[CoordSystem]

    fp_type = par.parval_from_str("fp_type")
    fp_type_alias = "DOUBLE" if fp_type == "float" else "REAL"
    loop_body = ccg.c_codegen(
        [
            rfm.xxSph[0],
            rfm.detgammahat,
        ],
        [
            "const DOUBLE r",
            "const DOUBLE sqrtdetgamma",
        ],
        include_braces=False,
        fp_type_alias=fp_type_alias,
    )

    loop_body += r"""
if(r < integration_radius) {
  const DOUBLE gf_of_x = in_gf[IDX4(gf_index, i0, i1, i2)];
  const DOUBLE dV = sqrtdetgamma * dxx0 * dxx1 * dxx2;
  squared_sum += gf_of_x * gf_of_x * dV;
  volume_sum  += dV;
} // END if(r < integration_radius)
"""
    body = r"""
  // Unpack grid parameters assuming a single grid
  const int grid = 0;
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"

  // Define reference metric grid
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];

  // Set summation variables to compute l2-norm
  DOUBLE squared_sum = 0.0;
  DOUBLE volume_sum  = 0.0;
"""

    body += lp.simple_loop(
        loop_body="\n" + loop_body,
        read_xxs=True,
        loop_region="interior",
        OMP_custom_pragma=r"#pragma omp parallel for reduction(+:squared_sum,volume_sum)",
    )

    body += r"""
  // Compute and output the log of the l2-norm.
  return log10(1e-16 + sqrt(squared_sum / volume_sum));  // 1e-16 + ... avoids log10(0)
"""

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        CoordSystem_for_wrapper_func="",
        name=name,
        params=params,
        include_CodeParameters_h=False,  # set_CodeParameters.h is manually included after the declaration of params_struct *restrict params
        body=body,
    )
