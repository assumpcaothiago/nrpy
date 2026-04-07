"""Bind standard BHaH griddata storage into the BHaHAHA interp_src interface."""

import nrpy.c_function as cfc


def register_CFunction_counterexample2_bind_griddata_to_interp_src() -> None:
    """Expose griddata[0].xx and auxevol_gfs through commondata->interp_src_*."""
    cfc.register_CFunction(
        includes=["BHaH_defines.h"],
        desc="Bind the standard BHaH griddata storage into the existing interp_src interface.",
        cfunc_type="void",
        name="counterexample2_bind_griddata_to_interp_src",
        params="commondata_struct *restrict commondata, griddata_struct *restrict griddata",
        include_CodeParameters_h=False,
        body=r"""
  params_struct *restrict params = &griddata[0].params;

  commondata->interp_src_Nxx0 = params->Nxx0;
  commondata->interp_src_Nxx1 = params->Nxx1;
  commondata->interp_src_Nxx2 = params->Nxx2;
  commondata->interp_src_Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  commondata->interp_src_Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  commondata->interp_src_Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  commondata->interp_src_dxx0 = params->dxx0;
  commondata->interp_src_dxx1 = params->dxx1;
  commondata->interp_src_dxx2 = params->dxx2;
  commondata->interp_src_invdxx0 = params->invdxx0;
  commondata->interp_src_invdxx1 = params->invdxx1;
  commondata->interp_src_invdxx2 = params->invdxx2;
  commondata->interp_src_r_theta_phi[0] = griddata[0].xx[0];
  commondata->interp_src_r_theta_phi[1] = griddata[0].xx[1];
  commondata->interp_src_r_theta_phi[2] = griddata[0].xx[2];
  commondata->interp_src_gfs = griddata[0].gridfuncs.auxevol_gfs;

  commondata->bcstruct_dxx0 = params->dxx0;
  commondata->bcstruct_dxx1 = params->dxx1;
  commondata->bcstruct_dxx2 = params->dxx2;
  commondata->bcstruct_Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  commondata->bcstruct_Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  commondata->bcstruct_Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
""",
    )
