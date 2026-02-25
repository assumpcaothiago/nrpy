# nrpy/infrastructures/manga/BHaH_evolve.py
"""
Generate function for time evolution.

Authors: Leonardo Rosa Werneck
         Thiago Assumpção
"""

from typing import Union, cast
from inspect import currentframe as cfr
from types import FrameType as FT

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_BHaH_evolve() -> Union[None, pcg.NRPyEnv_type]:
    """Register an interface function for time evolution.

    :return: None if in registration phase, else the updated NRPy environment.
    """

    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
    ]
    desc = "Perform spacetime evolution in BlackHoles@Home"
    cfunc_type = "void"
    name = "BHaH_evolve"
    params = "BHaH_struct *bhah_struct"
    include_CodeParameters_h = False
    body = r"""
    commondata_struct *commondata = bhah_struct->commondata;
    griddata_struct *griddata = bhah_struct->griddata;
    while(commondata->time < commondata->t_final) {
      // Only time evolution, since diagnostics are directly handled by MaNGa
      MoL_step_forward_in_time(commondata, griddata);
    }
"""

    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=include_CodeParameters_h,
        body=body,
    )

    return pcg.NRPyEnv()
