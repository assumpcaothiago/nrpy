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

@note:
The function writes in the following order:
1. Five integers:
   - NRPYELL_Nxx_plus_2NGHOSTS0
   - NRPYELL_Nxx_plus_2NGHOSTS1
   - NRPYELL_Nxx_plus_2NGHOSTS2
   - NRPYELL_NGHOSTS
   - NRPYELL_TOTAL_PTS
2. Six REALs:
   - NRPYELL_AMAX, NRPYELL_bScale, NRPYELL_SINHWAA,
     NRPYELL_dxx0, NRPYELL_dxx1, NRPYELL_dxx2
3. Three coordinate arrays (of sizes NRPYELL_Nxx_plus_2NGHOSTS0,
   NRPYELL_Nxx_plus_2NGHOSTS1, NRPYELL_Nxx_plus_2NGHOSTS2)
4. Two auxiliary evolution arrays (rho and P), each of size NRPYELL_TOTAL_PTS
5. Two evolution grid function arrays (psi_minus_one and alphaconf_minus_one),
   each of size NRPYELL_TOTAL_PTS
"""

    cfunc_type = "void"
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
    REAL *restrict auxevol_gfs = griddata[0].gridfuncs.auxevol_gfs;

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

    // Set pointers to arrays of coordinates xx[3]
    const REAL *restrict NRPYELL_xx0 = xx[0];
    const REAL *restrict NRPYELL_xx1 = xx[1];
    const REAL *restrict NRPYELL_xx2 = xx[2];

    // Set pointers to each auxevol grid function array of size NRPYELL_TOTAL_PTS
    const REAL *NRPYELL_rho = &auxevol_gfs[IDX4(RHOGF, 0, 0, 0)];
    const REAL *NRPYELL_P = &auxevol_gfs[IDX4(PGF, 0, 0, 0)];

    // Set pointers to each evol grid function array of size NRPYELL_TOTAL_PTS
    const REAL *NRPYELL_psi_minus_one = &y_n_gfs[IDX4(PSIGF, 0, 0, 0)];
    const REAL *NRPYELL_alphaconf_minus_one = &y_n_gfs[IDX4(ALPHACONFGF, 0, 0, 0)];

    // Open file for binary writing
    FILE *restrict fp = fopen("NRPYELL_solution.bin", "wb");
    if (fp == NULL) {
        perror("Error opening file for writing");
        exit(1);
    }

    // Write integer quantities
    if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS0, sizeof(NRPYELL_Nxx_plus_2NGHOSTS0), 1, fp) != 1) {
        perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS0");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS1, sizeof(NRPYELL_Nxx_plus_2NGHOSTS1), 1, fp) != 1) {
        perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS1");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_Nxx_plus_2NGHOSTS2, sizeof(NRPYELL_Nxx_plus_2NGHOSTS2), 1, fp) != 1) {
        perror("Error writing NRPYELL_Nxx_plus_2NGHOSTS2");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_NGHOSTS, sizeof(NRPYELL_NGHOSTS), 1, fp) != 1) {
        perror("Error writing NRPYELL_NGHOSTS");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_TOTAL_PTS, sizeof(NRPYELL_TOTAL_PTS), 1, fp) != 1) {
        perror("Error writing NRPYELL_TOTAL_PTS");
        fclose(fp);
        exit(1);
    }

    // Write double quantities
    if (fwrite(&NRPYELL_AMAX, sizeof(NRPYELL_AMAX), 1, fp) != 1) {
        perror("Error writing NRPYELL_AMAX");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_bScale, sizeof(NRPYELL_bScale), 1, fp) != 1) {
        perror("Error writing NRPYELL_bScale");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_SINHWAA, sizeof(NRPYELL_SINHWAA), 1, fp) != 1) {
        perror("Error writing NRPYELL_SINHWAA");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_dxx0, sizeof(NRPYELL_dxx0), 1, fp) != 1) {
        perror("Error writing NRPYELL_dxx0");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_dxx1, sizeof(NRPYELL_dxx1), 1, fp) != 1) {
        perror("Error writing NRPYELL_dxx1");
        fclose(fp);
        exit(1);
    }
    if (fwrite(&NRPYELL_dxx2, sizeof(NRPYELL_dxx2), 1, fp) != 1) {
        perror("Error writing NRPYELL_dxx2");
        fclose(fp);
        exit(1);
    }

    // Write coordinate arrays (sizes are NRPYELL_Nxx_plus_2NGHOSTS0, NRPYELL_Nxx_plus_2NGHOSTS1, NRPYELL_Nxx_plus_2NGHOSTS2)
    if (fwrite(NRPYELL_xx0, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS0, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS0) {
        perror("Error writing NRPYELL_xx0");
        fclose(fp);
        exit(1);
    }
    if (fwrite(NRPYELL_xx1, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS1, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS1) {
        perror("Error writing NRPYELL_xx1");
        fclose(fp);
        exit(1);
    }
    if (fwrite(NRPYELL_xx2, sizeof(REAL), NRPYELL_Nxx_plus_2NGHOSTS2, fp) != (size_t)NRPYELL_Nxx_plus_2NGHOSTS2) {
        perror("Error writing NRPYELL_xx2");
        fclose(fp);
        exit(1);
    }

    // Write auxiliary evolution arrays (each of size NRPYELL_TOTAL_PTS)
    if (fwrite(NRPYELL_rho, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
        perror("Error writing NRPYELL_rho");
        fclose(fp);
        exit(1);
    }
    if (fwrite(NRPYELL_P, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
        perror("Error writing NRPYELL_P");
        fclose(fp);
        exit(1);
    }

    // Write evolution grid function arrays (each of size NRPYELL_TOTAL_PTS)
    if (fwrite(NRPYELL_psi_minus_one, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
        perror("Error writing NRPYELL_psi_minus_one");
        fclose(fp);
        exit(1);
    }
    if (fwrite(NRPYELL_alphaconf_minus_one, sizeof(REAL), NRPYELL_TOTAL_PTS, fp) != (size_t)NRPYELL_TOTAL_PTS) {
        perror("Error writing NRPYELL_alphaconf_minus_one");
        fclose(fp);
        exit(1);
    }

    fclose(fp);
    printf("NRPYELL: FINISHED WRITING 'NRPYELL_solution.bin'\n");
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


def register_CFunction_read_NRPYELL_binary() -> Union[None, pcg.NRPyEnv_type]:
    """
    Register the C function that reads minimal NRPyElliptic data to binary file.

    :return: None if in registration phase, else the updated NRPy environment.
    """
    if pcg.pcg_registration_phase():
        pcg.register_func_call(f"{__name__}.{cast(FT, cfr()).f_code.co_name}", locals())
        return None

    includes = ["BHaH_defines.h"]

    prefunc = ""

    desc = r"""Reads minimal NRPyElliptic data from binary file 'NRPYELL_solution.bin'.

@note:
The function reads in the following data (in this order):
  1. Five integers:
     - NRPYELL_Nxx_plus_2NGHOSTS0
     - NRPYELL_Nxx_plus_2NGHOSTS1
     - NRPYELL_Nxx_plus_2NGHOSTS2
     - NRPYELL_NGHOSTS
     - NRPYELL_TOTAL_PTS
  2. Six REALs:
     - NRPYELL_AMAX, NRPYELL_bScale, NRPYELL_SINHWAA,
       NRPYELL_dxx0, NRPYELL_dxx1, NRPYELL_dxx2
  3. Three coordinate arrays (of sizes NRPYELL_Nxx_plus_2NGHOSTS0,
     NRPYELL_Nxx_plus_2NGHOSTS1, and NRPYELL_Nxx_plus_2NGHOSTS2 respectively)
  4. Two auxiliary evolution arrays (rho and P), each of size NRPYELL_TOTAL_PTS
  5. Two evolution grid function arrays (psi_minus_one and alphaconf_minus_one),
     each of size NRPYELL_TOTAL_PTS.

All function parameters are pointers that will be modified (or allocated) by the function.

@param NRPYELL_Nxx_plus_2NGHOSTS0 Pointer to an int that will hold the first coordinate size.
@param NRPYELL_Nxx_plus_2NGHOSTS1 Pointer to an int that will hold the second coordinate size.
@param NRPYELL_Nxx_plus_2NGHOSTS2 Pointer to an int that will hold the third coordinate size.
@param NRPYELL_NGHOSTS          Pointer to an int for the number of ghost zones.
@param NRPYELL_TOTAL_PTS         Pointer to an int for the total number of grid points.
@param NRPYELL_AMAX              Pointer to a REAL to store AMAX.
@param NRPYELL_bScale            Pointer to a REAL to store bScale.
@param NRPYELL_SINHWAA           Pointer to a REAL to store SINHWAA.
@param NRPYELL_dxx0              Pointer to a REAL to store dxx0.
@param NRPYELL_dxx1              Pointer to a REAL to store dxx1.
@param NRPYELL_dxx2              Pointer to a REAL to store dxx2.
@param NRPYELL_xx0               Pointer to a REAL* that will be allocated for coordinate array 0.
@param NRPYELL_xx1               Pointer to a REAL* that will be allocated for coordinate array 1.
@param NRPYELL_xx2               Pointer to a REAL* that will be allocated for coordinate array 2.
@param NRPYELL_rho               Pointer to a REAL* that will be allocated for the aux evolution array rho.
@param NRPYELL_P                 Pointer to a REAL* that will be allocated for the aux evolution array P.
@param NRPYELL_psi_minus_one     Pointer to a REAL* that will be allocated for the evolution array psi_minus_one.
@param NRPYELL_alphaconf_minus_one Pointer to a REAL* that will be allocated for the evolution array alphaconf_minus_one.

All function parameters are pointers that will be modified (or allocated) by the function.
"""

    cfunc_type = "void"
    name = "read_NRPYELL_binary"
    params = r"""
int * restrict NRPYELL_Nxx_plus_2NGHOSTS0,
int * restrict NRPYELL_Nxx_plus_2NGHOSTS1,
int * restrict NRPYELL_Nxx_plus_2NGHOSTS2,
int * restrict NRPYELL_NGHOSTS,
int * restrict NRPYELL_TOTAL_PTS,
REAL * restrict NRPYELL_AMAX,
REAL * restrict NRPYELL_bScale,
REAL * restrict NRPYELL_SINHWAA,
REAL * restrict NRPYELL_dxx0,
REAL * restrict NRPYELL_dxx1,
REAL * restrict NRPYELL_dxx2,
REAL ** restrict NRPYELL_xx0,
REAL ** restrict NRPYELL_xx1,
REAL ** restrict NRPYELL_xx2,
REAL ** restrict NRPYELL_rho,
REAL ** restrict NRPYELL_P,
REAL ** restrict NRPYELL_psi_minus_one,
REAL ** restrict NRPYELL_alphaconf_minus_one
"""

    body = r"""
    FILE * restrict fp = fopen("NRPYELL_solution.bin", "rb");
    if (fp == NULL) {
        perror("Error opening file for reading");
        exit(1);
    }

    // Read integer quantities
    if (fread(NRPYELL_Nxx_plus_2NGHOSTS0, sizeof(int), 1, fp) != 1) {
        perror("Error reading NRPYELL_Nxx_plus_2NGHOSTS0");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_Nxx_plus_2NGHOSTS1, sizeof(int), 1, fp) != 1) {
        perror("Error reading NRPYELL_Nxx_plus_2NGHOSTS1");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_Nxx_plus_2NGHOSTS2, sizeof(int), 1, fp) != 1) {
        perror("Error reading NRPYELL_Nxx_plus_2NGHOSTS2");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_NGHOSTS, sizeof(int), 1, fp) != 1) {
        perror("Error reading NRPYELL_NGHOSTS");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_TOTAL_PTS, sizeof(int), 1, fp) != 1) {
        perror("Error reading NRPYELL_TOTAL_PTS");
        fclose(fp);
        exit(1);
    }

    // Read double quantities
    if (fread(NRPYELL_AMAX, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_AMAX");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_bScale, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_bScale");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_SINHWAA, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_SINHWAA");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_dxx0, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_dxx0");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_dxx1, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_dxx1");
        fclose(fp);
        exit(1);
    }
    if (fread(NRPYELL_dxx2, sizeof(REAL), 1, fp) != 1) {
        perror("Error reading NRPYELL_dxx2");
        fclose(fp);
        exit(1);
    }

    // Allocate and read coordinate arrays
    *NRPYELL_xx0 = malloc((*NRPYELL_Nxx_plus_2NGHOSTS0) * sizeof(REAL));
    if (*NRPYELL_xx0 == NULL) {
        perror("Memory allocation failed for NRPYELL_xx0");
        fclose(fp);
        exit(1);
    }
    *NRPYELL_xx1 = malloc((*NRPYELL_Nxx_plus_2NGHOSTS1) * sizeof(REAL));
    if (*NRPYELL_xx1 == NULL) {
        perror("Memory allocation failed for NRPYELL_xx1");
        fclose(fp);
        exit(1);
    }
    *NRPYELL_xx2 = malloc((*NRPYELL_Nxx_plus_2NGHOSTS2) * sizeof(REAL));
    if (*NRPYELL_xx2 == NULL) {
        perror("Memory allocation failed for NRPYELL_xx2");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_xx0, sizeof(REAL), *NRPYELL_Nxx_plus_2NGHOSTS0, fp) != (size_t)*NRPYELL_Nxx_plus_2NGHOSTS0) {
        perror("Error reading coordinate array xx0");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_xx1, sizeof(REAL), *NRPYELL_Nxx_plus_2NGHOSTS1, fp) != (size_t)*NRPYELL_Nxx_plus_2NGHOSTS1) {
        perror("Error reading coordinate array xx1");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_xx2, sizeof(REAL), *NRPYELL_Nxx_plus_2NGHOSTS2, fp) != (size_t)*NRPYELL_Nxx_plus_2NGHOSTS2) {
        perror("Error reading coordinate array xx2");
        fclose(fp);
        exit(1);
    }

    // Allocate and read auxiliary evolution arrays (rho and P)
    *NRPYELL_rho = malloc((*NRPYELL_TOTAL_PTS) * sizeof(REAL));
    if (*NRPYELL_rho == NULL) {
        perror("Memory allocation failed for NRPYELL_rho");
        fclose(fp);
        exit(1);
    }
    *NRPYELL_P = malloc((*NRPYELL_TOTAL_PTS) * sizeof(REAL));
    if (*NRPYELL_P == NULL) {
        perror("Memory allocation failed for NRPYELL_P");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_rho, sizeof(REAL), *NRPYELL_TOTAL_PTS, fp) != (size_t)*NRPYELL_TOTAL_PTS) {
        perror("Error reading auxiliary evolution array rho");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_P, sizeof(REAL), *NRPYELL_TOTAL_PTS, fp) != (size_t)*NRPYELL_TOTAL_PTS) {
        perror("Error reading auxiliary evolution array P");
        fclose(fp);
        exit(1);
    }

    // Allocate and read evolution grid function arrays (psi_minus_one and alphaconf_minus_one)
    *NRPYELL_psi_minus_one = malloc((*NRPYELL_TOTAL_PTS) * sizeof(REAL));
    if (*NRPYELL_psi_minus_one == NULL) {
        perror("Memory allocation failed for NRPYELL_psi_minus_one");
        fclose(fp);
        exit(1);
    }
    *NRPYELL_alphaconf_minus_one = malloc((*NRPYELL_TOTAL_PTS) * sizeof(REAL));
    if (*NRPYELL_alphaconf_minus_one == NULL) {
        perror("Memory allocation failed for NRPYELL_alphaconf_minus_one");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_psi_minus_one, sizeof(REAL), *NRPYELL_TOTAL_PTS, fp) != (size_t)*NRPYELL_TOTAL_PTS) {
        perror("Error reading evolution array psi_minus_one");
        fclose(fp);
        exit(1);
    }
    if (fread(*NRPYELL_alphaconf_minus_one, sizeof(REAL), *NRPYELL_TOTAL_PTS, fp) != (size_t)*NRPYELL_TOTAL_PTS) {
        perror("Error reading evolution array alphaconf_minus_one");
        fclose(fp);
        exit(1);
    }

    fclose(fp);
    printf("NRPYELL: FINISHED READING 'NRPYELL_solution.bin'\n");
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
