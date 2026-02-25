# nrpy/infrastructures/manga/BHaH_checkpoint_and_diagnostics.py
"""
Construct C function to write checkpoint file and diagnostics using custom file names.

Author: Thiago Assumpcao
        assumpcaothiago **at** gmail **dot* com
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast
import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_BHaH_checkpoint_and_diagnostics() -> (
    Union[None, pcg.NRPyEnv_type]
):
    """
    Register function to write checkpoint file and diagnostics using custom file names.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h", "BHaH_function_prototypes.h"]

    desc = "write checkpoint file and diagnostics using custom file names."

    cfunc_type = "void"
    name = "BHaH_checkpoint_and_diagnostics"
    params = "BHaH_struct *bhahstruct, char *filename"

    body = r"""
// Step 1: Define local structures
commondata_struct *commondata = bhahstruct->commondata;
griddata_struct *griddata = bhahstruct->griddata;

// TODO: Step 2: Trigger checkpoint and diagnostics condition

// Step 3.a: Write checkpoint using default function, which creates a file named 'checkpoint-conv_factor1.00.dat'
write_checkpoint(commondata, griddata);

// Step 3.b: Rename file to the parameter 'filename'
const char *default_filename = "checkpoint-conv_factor1.00.dat";
if (rename(default_filename, filename) != 0) {
perror("BHaH_checkpoint_and_diagnostics: ERROR renaming checkpoint file");
exit(EXIT_FAILURE);
}

// Step 4.a: Output diagnostics
diagnostics(commondata, griddata);

// TODO: Step 4.b: Rename diagnostics files

"""

    cfc.register_CFunction(
        include_CodeParameters_h=False,
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        body=body,
        enable_simd=False,
    )
    return pcg.NRPyEnv()
