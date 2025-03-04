"""
Register C function for that writes minimal NRPyElliptic data to binary file.

Author: Thiago Assumpcao (adapted to 2d)
"""

from inspect import currentframe as cfr
from types import FrameType as FT
from typing import Union, cast

import nrpy.c_function as cfc
import nrpy.helpers.parallel_codegen as pcg


def register_CFunction_write_NRPYELL_binary() -> Union[None, pcg.NRPyEnv_type]:
    """
    Register the C function that writes minimal NRPyElliptic data to binary file.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]

    prefunc = ""

    desc = r"""Writes minimal NRPyElliptic data to binary file

@param griddata - Grid data struct.

@return - Error code indicating success or error.
"""

    cfunc_type = "int"
    name = "write_NRPYELL_binary"
    params = "griddata_struct *restrict griddata"

    body = r"""
  // Define some useful quantities to set up data output
  params_struct *restrict params = &griddata[0].params;
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[0].xx[ww];
  const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  REAL *restrict y_n_gfs = griddata[0].gridfuncs.y_n_gfs;

  // Set integers to be written to file
  const int NRPYELL_Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  const int NRPYELL_Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  const int NRPYELL_Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;
  const int NRPYELL_NGHOSTS = NGHOSTS;
  const int NRPYELL_TOTAL_PTS = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;

  // Set doubles to be written to file
  const REAL NRPYELL_AMAX = params->AMAX;
  const REAL NRPYELL_bScale = params->bScale;
  const REAL NRPYELL_SINHWAA = params->SINHWAA;
  const REAL NRPYELL_dxx0 = params->dxx0;
  const REAL NRPYELL_dxx1 = params->dxx1;
  const REAL NRPYELL_dxx2 = params->dxx2;

  // Set pointers to each grid function array of size NRPYELL_TOTAL_PTS
  const REAL *NRPYELL_psi_minus_one = &y_n_gfs[IDX4(PSIGF, 0, 0, 0)];
  const REAL *NRPYELL_alphaconf_minus_one = &y_n_gfs[IDX4(ALPHACONFGF, 0, 0, 0)];

  // Set pointers to arrays of coordinates xx[3]
  const REAL *restrict NRPYELL_xx0 = xx[0];
  const REAL *restrict NRPYELL_xx1 = xx[1];
  const REAL *restrict NRPYELL_xx2 = xx[2];

  /* Open file for binary writing */
  FILE *fp = fopen("NRPYELL_solution.bin", "wb");
  if (fp == NULL) {
    perror("Error opening file for writing");
    return -1;
  }

  /* Write integer quantities */
  if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS0, sizeof(NRPYELL_Nxx_plus_2NGHOSTS0), 1, fp) != 1) {
    perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS0");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS1, sizeof(NRPYELL_Nxx_plus_2NGHOSTS1), 1, fp) != 1) {
    perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS1");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS2, sizeof(NRPYELL_Nxx_plus_2NGHOSTS2), 1, fp) != 1) {
    perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS2");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_NGHOSTS, sizeof(NRPYELL_NGHOSTS), 1, fp) != 1) {
    perror("Error writing NRPYELL_NGHOSTS");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_TOTAL_PTS, sizeof(NRPYELL_TOTAL_PTS), 1, fp) != 1) {
    perror("Error writing NRPYELL_TOTAL_PTS");
    fclose(fp);
    return -1;
  }

  /* Write double quantities */
  if (fwrite(&NRPYELL_AMAX, sizeof(NRPYELL_AMAX), 1, fp) != 1) {
    perror("Error writing NRPYELL_AMAX");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_bScale, sizeof(NRPYELL_bScale), 1, fp) != 1) {
    perror("Error writing NRPYELL_bScale");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_SINHWAA, sizeof(NRPYELL_SINHWAA), 1, fp) != 1) {
    perror("Error writing NRPYELL_SINHWAA");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_dxx0, sizeof(NRPYELL_dxx0), 1, fp) != 1) {
    perror("Error writing NRPYELL_dxx0");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_dxx1, sizeof(NRPYELL_dxx1), 1, fp) != 1) {
    perror("Error writing NRPYELL_dxx1");
    fclose(fp);
    return -1;
  }
  if (fwrite(&NRPYELL_dxx2, sizeof(NRPYELL_dxx2), 1, fp) != 1) {
    perror("Error writing NRPYELL_dxx2");
    fclose(fp);
    return -1;
  }

  /* Write grid function arrays (each of size NRPYELL_TOTAL_PTS) */
  if (fwrite(NRPYELL_psi_minus_one, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
    perror("Error writing NRPYELL_psi_minus_one");
    fclose(fp);
    return -1;
  }
  if (fwrite(NRPYELL_alphaconf_minus_one, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
    perror("Error writing NRPYELL_alphaconf_minus_one");
    fclose(fp);
    return -1;
  }

  /* Write coordinate arrays (sizes are NRPYELL_Nxx_plus_2NGHOSTS0, NRPYELL_Nxx_plus_2NGHOSTS1, NRPYELL_Nxx_plus_2NGHOSTS2) */
  if (fwrite(NRPYELL_xx0, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS0, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS0) {
    perror("Error writing NRPYELL_xx0");
    fclose(fp);
    return -1;
  }
  if (fwrite(NRPYELL_xx1, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS1, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS1) {
    perror("Error writing NRPYELL_xx1");
    fclose(fp);
    return -1;
  }
  if (fwrite(NRPYELL_xx2, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS2, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS2) {
    perror("Error writing NRPYELL_xx2");
    fclose(fp);
    return -1;
  }

  fclose(fp);
  printf("NRPYELL: FINISHED WRITING 'NRPYELL_solution.bin'\n");
  return 0;
"""

    postfunc = ""

    cfc.register_CFunction(
        subdirectory="",
        includes=includes,
        prefunc=prefunc,
        desc=desc,
        cfunc_type=cfunc_type,
        name=name,
        params=params,
        include_CodeParameters_h=False,
        body=body,
        postfunc=postfunc,
    )
    return cast(pcg.NRPyEnv_type, pcg.NRPyEnv())
