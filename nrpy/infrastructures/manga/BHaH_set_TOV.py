# nrpy/infrastructures/manga/BHaH_set_TOV.py
"""
Generate function to set TOV solution for MaNGa.

Author: Thiago Assumpção
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_BHaH_set_TOV() -> Union[None, pcg.NRPyEnv_type]:
    """
    Register function to set TOV solution for MaNGa.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "Set TOV solution for MaNGa."
    cfunc_type = "void"
    name = "BHaH_set_TOV"
    params = r"""BHaH_struct *bhahstruct, const int num_radial_pts,
    REAL *restrict r_axis, REAL *restrict rho_baryon, REAL *restrict pressure"""

    body = r"""
  // Unpack data from bhahstruct
  commondata_struct *commondata = bhahstruct->commondata;
  griddata_struct *griddata = bhahstruct->griddata;

  manga_radial_initial_data(commondata, griddata, num_radial_pts, r_axis, rho_baryon, pressure);
"""

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
