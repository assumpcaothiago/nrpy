"""Register the minimal gridfunction and struct footprint for the diagnostic."""

import nrpy.grid as gri
import nrpy.params as par
from nrpy.infrastructures import BHaH


def register_gridfunctions() -> None:
    """Register the AUXEVOL gridfunctions used by the naive derivative diagnostic."""
    _ = gri.register_gridfunctions_for_single_rank2(
        "aDD", symmetry="sym01", group="AUXEVOL", gf_array_name="auxevol_gfs"
    )
    _ = gri.register_gridfunctions_for_single_rank2(
        "hDD", symmetry="sym01", group="AUXEVOL", gf_array_name="auxevol_gfs"
    )
    _ = gri.register_gridfunctions_for_single_rankN(
        "partial_D_hDD",
        rank=3,
        symmetry="sym12",
        group="AUXEVOL",
        gf_array_name="auxevol_gfs",
    )
    _ = gri.register_gridfunctions_for_single_rank1(
        "partial_D_WW", group="AUXEVOL", gf_array_name="auxevol_gfs"
    )
    _ = gri.register_gridfunctions("trK", group="AUXEVOL", gf_array_name="auxevol_gfs")[
        0
    ]
    _ = gri.register_gridfunctions("WW", group="AUXEVOL", gf_array_name="auxevol_gfs")[
        0
    ]

    # numerical_grids_and_timestep initializes these fields even in a diagnostic-only
    # executable, so the standard commondata members need to exist.
    _ = par.CodeParameter(
        "int",
        __name__,
        "nn_0",
        add_to_parfile=False,
        add_to_set_CodeParameters_h=True,
        commondata=True,
    )
    _ = par.CodeParameter(
        "int",
        __name__,
        "nn",
        add_to_parfile=False,
        add_to_set_CodeParameters_h=True,
        commondata=True,
    )
    _ = par.CodeParameter(
        "REAL",
        __name__,
        "dt",
        add_to_parfile=False,
        add_to_set_CodeParameters_h=True,
        commondata=True,
    )
    _ = par.CodeParameter(
        "REAL",
        __name__,
        "t_0",
        add_to_parfile=False,
        add_to_set_CodeParameters_h=True,
        commondata=True,
    )
    _ = par.CodeParameter(
        "REAL",
        __name__,
        "time",
        add_to_parfile=False,
        add_to_set_CodeParameters_h=True,
        commondata=True,
    )

    butcher_dict = BHaH.MoLtimestepping.rk_butcher_table_dictionary.generate_Butcher_tables()
    BHaH.MoLtimestepping.BHaH_defines.register_BHaH_defines_h(
        butcher_dict, "Euler"
    )
    BHaH.griddata_commondata.register_griddata_commondata(
        __name__,
        "MoL_gridfunctions_struct gridfuncs",
        "MoL gridfunctions",
    )
    BHaH.griddata_commondata.register_griddata_commondata(
        __name__,
        "bhahaha_params_and_data_struct *restrict bhahaha_params_and_data",
        "input parameters and data set by the diagnostic",
        is_commondata=True,
    )
