"""Definitions and error codes for the Counterexample 2 minimal diagnostic."""

from nrpy.infrastructures import BHaH

counterexample2_error_code_msg_tuples = [
    (
        "COUNTEREXAMPLE2_OUTPUT_IO_ERROR",
        "Counterexample 2 diagnostic: failed to create or write one or more output files.",
    ),
    (
        "COUNTEREXAMPLE2_CARTESIAN_DOMAIN_TOO_SMALL",
        "Counterexample 2 diagnostic: source radial extent is too small for the requested Cartesian half-width.",
    ),
    (
        "COUNTEREXAMPLE2_CARTESIAN_INTERP_FAILURE",
        "Counterexample 2 diagnostic: 3D Cartesian interpolation failed.",
    ),
    (
        "COUNTEREXAMPLE2_AUXEVOL_MALLOC_ERROR",
        "Counterexample 2 diagnostic: failed to allocate auxevol gridfunction storage.",
    ),
    (
        "COUNTEREXAMPLE2_GRIDDATA_MALLOC_ERROR",
        "Counterexample 2 diagnostic: failed to allocate griddata storage.",
    ),
    (
        "COUNTEREXAMPLE2_TEMP_MALLOC_ERROR",
        "Counterexample 2 diagnostic: failed to allocate temporary Cartesian sampling storage.",
    ),
]


def register_error_codes() -> None:
    """Register Counterexample 2-specific error codes with BHaHAHA's error handler."""
    existing = {
        name for name, _ in BHaH.BHaHAHA.error_message.error_code_msg_tuples_list
    }
    for error_tuple in counterexample2_error_code_msg_tuples:
        if error_tuple[0] not in existing:
            BHaH.BHaHAHA.error_message.error_code_msg_tuples_list.append(error_tuple)
            existing.add(error_tuple[0])


def register_BHaH_defines(cartesian_half_width: float) -> None:
    """Register BHaH_defines.h contributions for the Cartesian diagnostic outputs."""
    BHaH.BHaH_defines_h.register_BHaH_defines(
        __name__,
        f"""
#define COUNTEREXAMPLE2_CARTESIAN_HALF_WIDTH {cartesian_half_width:.17g}
#define COUNTEREXAMPLE2_CART_EPS_FACTOR 0.25
#define COUNTEREXAMPLE2_NUM_CART_INTERP_GFS 3
#define COUNTEREXAMPLE2_CART_LINE_SAMPLES 257
#define COUNTEREXAMPLE2_CART_PLANE_SAMPLES 257
enum {{
  COUNTEREXAMPLE2_CART_X_LINE = 0,
  COUNTEREXAMPLE2_CART_Y_LINE,
  COUNTEREXAMPLE2_CART_Z_LINE,
  COUNTEREXAMPLE2_CART_XY_PLANE,
  COUNTEREXAMPLE2_CART_XZ_PLANE,
  COUNTEREXAMPLE2_CART_YZ_PLANE
}};
""",
    )
