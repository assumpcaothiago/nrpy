"""Allocation helpers for the Counterexample 2 minimal diagnostic."""

import nrpy.c_function as cfc


def register_CFunction_counterexample2_allocate_auxevol_gfs() -> None:
    """Allocate and zero the standard AUXEVOL storage used for the diagnostic."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h", "string.h"],
        desc="Allocate and zero auxevol gridfunction storage for the diagnostic grid.",
        cfunc_type="int",
        name="counterexample2_allocate_auxevol_gfs",
        params="griddata_struct *restrict griddata",
        include_CodeParameters_h=False,
        body=r"""
  params_struct *restrict params = &griddata[0].params;
  const size_t ntot = (size_t)params->Nxx_plus_2NGHOSTS0 * (size_t)params->Nxx_plus_2NGHOSTS1 * (size_t)params->Nxx_plus_2NGHOSTS2;
  BHAH_MALLOC(griddata[0].gridfuncs.auxevol_gfs, sizeof(REAL) * ntot * NUM_AUXEVOL_GFS);
  if (griddata[0].gridfuncs.auxevol_gfs == NULL)
    return COUNTEREXAMPLE2_AUXEVOL_MALLOC_ERROR;
  memset(griddata[0].gridfuncs.auxevol_gfs, 0, sizeof(REAL) * ntot * NUM_AUXEVOL_GFS);
  return BHAHAHA_SUCCESS;
""",
    )
