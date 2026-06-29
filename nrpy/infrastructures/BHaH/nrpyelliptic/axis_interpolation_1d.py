"""
Register final-state 1D axis interpolation output for NRPyElliptic.

The generated C function samples the evolved field ``uu`` on a user-selected
Cartesian x- or z-axis line by converting target Cartesian points to the active
coordinate system and calling the general 3D uniform-grid Lagrange interpolator.
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg
import nrpy.params as par


def register_CFunction_axis_interpolation_1d_output(
    default_axis: str = "x",
    default_num_points: int = 1024,
    default_x_min: float = -10.0,
    default_x_max: float = 10.0,
    default_z_min: float = -10.0,
    default_z_max: float = 10.0,
    default_offset: float = 1.0e-6,
) -> Union[None, pcg.NRPyEnv_type]:
    """
    Register a generated C function that writes interpolated ``uu`` along one axis.

    Runtime control is via commondata CodeParameters:
      - ``axis_interpolation_enable``
      - ``axis_interpolation_axis`` ("x" or "z")
      - ``axis_interpolation_num_points``
      - ``axis_interpolation_x_min``, ``axis_interpolation_x_max``
      - ``axis_interpolation_z_min``, ``axis_interpolation_z_max``
      - ``axis_interpolation_x_offset``, ``axis_interpolation_y_offset``,
        ``axis_interpolation_z_offset``

    :param default_axis: Default axis selector, "x" or "z".
    :param default_num_points: Default number of points on the output line.
    :param default_x_min: Default minimum x coordinate for x-axis output.
    :param default_x_max: Default maximum x coordinate for x-axis output.
    :param default_z_min: Default minimum z coordinate for z-axis output.
    :param default_z_max: Default maximum z coordinate for z-axis output.
    :param default_offset: Default off-axis Cartesian coordinate value.
    :return: None during parallel-codegen registration, else updated NRPyEnv.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    if default_axis not in ("x", "z"):
        raise ValueError(f'default_axis must be "x" or "z"; got {default_axis!r}')
    if default_num_points < 2:
        raise ValueError("default_num_points must be at least 2")

    _ = par.CodeParameter(
        "bool",
        __name__,
        "axis_interpolation_enable",
        True,
        commondata=True,
    )
    _ = par.CodeParameter(
        "char[10]",
        __name__,
        "axis_interpolation_axis",
        default_axis,
        commondata=True,
    )
    _ = par.CodeParameter(
        "int",
        __name__,
        "axis_interpolation_num_points",
        default_num_points,
        commondata=True,
    )
    for name, default in [
        ("axis_interpolation_x_min", default_x_min),
        ("axis_interpolation_x_max", default_x_max),
        ("axis_interpolation_z_min", default_z_min),
        ("axis_interpolation_z_max", default_z_max),
        ("axis_interpolation_x_offset", default_offset),
        ("axis_interpolation_y_offset", default_offset),
        ("axis_interpolation_z_offset", default_offset),
    ]:
        _ = par.CodeParameter("REAL", __name__, name, default, commondata=True)

    includes = [
        "BHaH_defines.h",
        "BHaH_function_prototypes.h",
        "math.h",
        "stdio.h",
        "stdlib.h",
        "string.h",
    ]
    desc = """
Write final-state 1D interpolation output for the NRPyElliptic evolved field uu.

The routine supports Cartesian x-axis and z-axis line outputs. For x-axis output,
the x coordinate varies over [axis_interpolation_x_min, axis_interpolation_x_max]
while y and z are fixed to axis_interpolation_y_offset and
axis_interpolation_z_offset. For z-axis output, z varies over
[axis_interpolation_z_min, axis_interpolation_z_max] while x and y are fixed to
axis_interpolation_x_offset and axis_interpolation_y_offset.

For each requested Cartesian point, the routine converts Cartesian coordinates
to the active grid coordinates using Cart_to_xx_and_nearest_i0i1i2(), then
interpolates uu using interpolation_3d_general__uniform_src_grid(). Output rows
contain x, y, z, and uu.
"""
    cfunc_type = "void"
    name = "axis_interpolation_1d_output"
    params = "const commondata_struct *restrict commondata, griddata_struct *restrict griddata"

    body = r"""
  if (!commondata->axis_interpolation_enable) {
    return;
  }

  const int num_points = commondata->axis_interpolation_num_points;
  if (num_points < 2) {
    fprintf(stderr, "axis_interpolation_1d_output: axis_interpolation_num_points must be >= 2; got %d\n", num_points);
    exit(1);
  }

  const int output_x_axis = strncmp(commondata->axis_interpolation_axis, "x", 10) == 0 ||
                            strncmp(commondata->axis_interpolation_axis, "X", 10) == 0;
  const int output_z_axis = strncmp(commondata->axis_interpolation_axis, "z", 10) == 0 ||
                            strncmp(commondata->axis_interpolation_axis, "Z", 10) == 0;
  if (!output_x_axis && !output_z_axis) {
    fprintf(stderr, "axis_interpolation_1d_output: axis_interpolation_axis must be \"x\" or \"z\"; got \"%s\"\n",
            commondata->axis_interpolation_axis);
    exit(1);
  }

  const REAL axis_min = output_x_axis ? commondata->axis_interpolation_x_min : commondata->axis_interpolation_z_min;
  const REAL axis_max = output_x_axis ? commondata->axis_interpolation_x_max : commondata->axis_interpolation_z_max;
  const char axis_char = output_x_axis ? 'x' : 'z';
  const REAL denom = (REAL)(num_points - 1);

  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    const params_struct *restrict params = &griddata[grid].params;
    REAL *restrict src_x0x1x2[3] = {griddata[grid].xx[0], griddata[grid].xx[1], griddata[grid].xx[2]};
    const REAL *restrict src_gf_ptrs[1] = {&griddata[grid].gridfuncs.y_n_gfs[IDX4Ppt(params, UUGF, 0)]};

    REAL(*restrict dst_x0x1x2)[3] = (REAL(*)[3])malloc(sizeof(REAL[3]) * (size_t)num_points);
    REAL *restrict x_out = (REAL *)malloc(sizeof(REAL) * (size_t)num_points);
    REAL *restrict y_out = (REAL *)malloc(sizeof(REAL) * (size_t)num_points);
    REAL *restrict z_out = (REAL *)malloc(sizeof(REAL) * (size_t)num_points);
    REAL *restrict uu_interp = (REAL *)malloc(sizeof(REAL) * (size_t)num_points);
    if (dst_x0x1x2 == NULL || x_out == NULL || y_out == NULL || z_out == NULL || uu_interp == NULL) {
      fprintf(stderr, "axis_interpolation_1d_output: allocation failure for %d points on grid %d\n", num_points, grid);
      free(dst_x0x1x2);
      free(x_out);
      free(y_out);
      free(z_out);
      free(uu_interp);
      exit(1);
    }

    for (int i = 0; i < num_points; i++) {
      const REAL axis_coord = axis_min + (axis_max - axis_min) * ((REAL)i) / denom;
      REAL xCart[3];
      if (output_x_axis) {
        xCart[0] = axis_coord;
        xCart[1] = commondata->axis_interpolation_y_offset;
        xCart[2] = commondata->axis_interpolation_z_offset;
      } else {
        xCart[0] = commondata->axis_interpolation_x_offset;
        xCart[1] = commondata->axis_interpolation_y_offset;
        xCart[2] = axis_coord;
      }
      int nearest_i0i1i2[3];
      Cart_to_xx_and_nearest_i0i1i2(params, xCart, dst_x0x1x2[i], nearest_i0i1i2);
      x_out[i] = xCart[0];
      y_out[i] = xCart[1];
      z_out[i] = xCart[2];
    }

    REAL *restrict dst_data[1] = {uu_interp};
    const int interp_error = interpolation_3d_general__uniform_src_grid(
        NGHOSTS, params->dxx0, params->dxx1, params->dxx2,
        params->Nxx_plus_2NGHOSTS0, params->Nxx_plus_2NGHOSTS1, params->Nxx_plus_2NGHOSTS2,
        1, src_x0x1x2, src_gf_ptrs, num_points, (const REAL(*)[3])dst_x0x1x2, dst_data);
    if (interp_error != 0) {
      fprintf(stderr,
              "axis_interpolation_1d_output: interpolation failed with error code %d on grid %d. "
              "Check requested %c-axis range and offsets.\n",
              interp_error, grid, axis_char);
      free(dst_x0x1x2);
      free(x_out);
      free(y_out);
      free(z_out);
      free(uu_interp);
      exit(1);
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "out1d-interp-uu-%c-%s-grid%02d-conv_factor%.2f-n%08d.txt",
             axis_char, params->CoordSystemName, grid, commondata->convergence_factor, commondata->nn);
    FILE *out = fopen(filename, "w");
    if (out == NULL) {
      fprintf(stderr, "axis_interpolation_1d_output: could not open %s for writing\n", filename);
      free(dst_x0x1x2);
      free(x_out);
      free(y_out);
      free(z_out);
      free(uu_interp);
      exit(1);
    }

    fprintf(out, "# time %.15e\n", commondata->time);
    fprintf(out, "# nn %d\n", commondata->nn);
    fprintf(out, "# axis %c\n", axis_char);
    fprintf(out, "# columns: x y z uu\n");
    for (int i = 0; i < num_points; i++) {
      fprintf(out, "%.15e %.15e %.15e %.15e\n", x_out[i], y_out[i], z_out[i], uu_interp[i]);
    }
    fclose(out);

    free(dst_x0x1x2);
    free(x_out);
    free(y_out);
    free(z_out);
    free(uu_interp);
  }
"""
    cfc.register_CFunction(
        includes=includes,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
    )
    return pcg.NRPyEnv()


if __name__ == "__main__":
    import doctest
    import sys

    results = doctest.testmod()

    if results.failed > 0:
        print(f"Doctest failed: {results.failed} of {results.attempted} test(s)")
        sys.exit(1)
    else:
        print(f"Doctest passed: All {results.attempted} test(s) passed")
