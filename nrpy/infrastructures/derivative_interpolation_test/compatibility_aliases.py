"""Compatibility aliases that map BHaHAHA interp_src enums onto AUXEVOL enums."""

from typing import List

from nrpy.infrastructures import BHaH


def _ordered_auxevol_names() -> List[str]:
    names: List[str] = []
    for i in range(3):
        for j in range(i, 3):
            names.append(f"aDD{i}{j}")
    for i in range(3):
        for j in range(i, 3):
            names.append(f"hDD{i}{j}")
    for k in range(3):
        for i in range(3):
            for j in range(i, 3):
                names.append(f"partial_D_hDD{k}{i}{j}")
    for i in range(3):
        names.append(f"partial_D_WW{i}")
    names.extend(["trK", "WW"])
    names.sort(key=str.upper)
    return names


def register_BHaH_defines() -> None:
    """Register macro aliases so existing interp_src routines operate on auxevol_gfs."""
    alias_lines = [
        "// Alias interp_src gridfunction names onto AUXEVOL enums so the existing",
        "// BHaHAHA interpolation-source routines can operate directly on auxevol_gfs.",
        "#undef NUM_INTERP_SRC_GFS",
        "#define NUM_INTERP_SRC_GFS NUM_AUXEVOL_GFS",
    ]
    for name in _ordered_auxevol_names():
        upper_name = name.upper()
        alias_lines.append(f"#define SRC_{upper_name}GF {upper_name}GF")
    BHaH.BHaH_defines_h.register_BHaH_defines(
        "zz_nrpy.infrastructures.derivative_interpolation_test.compatibility_aliases",
        "\n".join(alias_lines) + "\n",
    )
