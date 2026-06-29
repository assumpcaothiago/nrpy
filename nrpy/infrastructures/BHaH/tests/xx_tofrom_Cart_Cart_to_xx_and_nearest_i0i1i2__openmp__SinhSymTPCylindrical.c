#include "BHaH_defines.h"

/**
 * Given Cartesian point (x,y,z), this function unshifts the grid back to the origin to output the corresponding
 *             (xx0,xx1,xx2) and the "closest" (i0,i1,i2) for the given grid
 */
void Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTPCylindrical(const params_struct *restrict params, const REAL xCart[3], REAL xx[3],
                                                              int Cart_to_i0i1i2[3]) {
  // Set (Cartx, Carty, Cartz) relative to the global (as opposed to local) grid.
  //   This local grid may be offset from the origin by adjusting
  //   (Cart_originx, Cart_originy, Cart_originz) to nonzero values.
  REAL Cartx = xCart[0];
  REAL Carty = xCart[1];
  REAL Cartz = xCart[2];

  // Set the origin, (Cartx, Carty, Cartz) = (0, 0, 0), to the center of the local grid patch.
  Cartx -= params->Cart_originx;
  Carty -= params->Cart_originy;
  Cartz -= params->Cart_originz;
  {
    /*
     *  Original SymPy expressions:
     *  "[xx[0] = Piecewise((params->SINHWXY*asinh(sqrt(Cartx**2/2 + Carty**2/2 - params->bScaleXY**2/2 + sqrt(4*Carty**2*params->bScaleXY**2 +
     * (Cartx**2 + Carty**2 - params->bScaleXY**2)**2)/2)*sinh(1/params->SINHWXY)/params->AMPLXY), Cartx**2 + Carty**2 - params->bScaleXY**2 >= 0),
     * (params->SINHWXY*asinh(sqrt(2)*sqrt(Carty**2/(-Cartx**2 - Carty**2 + params->bScaleXY**2 + sqrt(4*Carty**2*params->bScaleXY**2 + (Cartx**2 +
     * Carty**2 - params->bScaleXY**2)**2)))*sinh(1/params->SINHWXY)*Abs(params->bScaleXY)/params->AMPLXY), True))]"
     *  "[xx[1] = Piecewise((atan2(Cartx*sqrt(Cartx**2/2 + Carty**2/2 - params->bScaleXY**2/2 + sqrt(4*Carty**2*params->bScaleXY**2 + (Cartx**2 +
     * Carty**2 - params->bScaleXY**2)**2)/2), Carty*sqrt(Cartx**2/2 + Carty**2/2 + params->bScaleXY**2/2 + sqrt(4*Carty**2*params->bScaleXY**2 +
     * (Cartx**2 + Carty**2 - params->bScaleXY**2)**2)/2)), Cartx**2 + Carty**2 - params->bScaleXY**2 >= 0),
     * (atan2(sqrt(2)*Cartx*sqrt(Carty**2/(-Cartx**2 - Carty**2 + params->bScaleXY**2 + sqrt(4*Carty**2*params->bScaleXY**2 + (Cartx**2 + Carty**2 -
     * params->bScaleXY**2)**2)))*Abs(params->bScaleXY), Carty*sqrt(2*Carty**2*params->bScaleXY**2/(-Cartx**2 - Carty**2 + params->bScaleXY**2 +
     * sqrt(4*Carty**2*params->bScaleXY**2 + (Cartx**2 + Carty**2 - params->bScaleXY**2)**2)) + params->bScaleXY**2)), True))]"
     *  "[xx[2] = params->SINHWZ*asinh(Cartz*sinh(1/params->SINHWZ)/params->AMPLZ)]"
     */
    const REAL tmp0 = ((params->bScaleXY) * (params->bScaleXY));
    const REAL tmp3 = ((Carty) * (Carty));
    const REAL tmp8 = sinh((1.0 / (params->SINHWXY))) / params->AMPLXY;
    const REAL tmp4 = ((Cartx) * (Cartx)) - tmp0 + tmp3;
    const REAL tmp5 = sqrt(4 * tmp0 * tmp3 + ((tmp4) * (tmp4)));
    const REAL tmp9 = tmp4 >= 0;
    const REAL tmp6 = (1.0 / 2.0) * ((Cartx) * (Cartx)) + (1.0 / 2.0) * tmp3 + (1.0 / 2.0) * tmp5;
    const REAL tmp10 = tmp3 / (-tmp4 + tmp5);
    const REAL tmp7 = sqrt(-1.0 / 2.0 * tmp0 + tmp6);
    const REAL tmp11 = M_SQRT2 * sqrt(tmp10) * fabs(params->bScaleXY);
    if (tmp9) {
      xx[0] = params->SINHWXY * asinh(tmp7 * tmp8);
    } else {
      xx[0] = params->SINHWXY * asinh(tmp11 * tmp8);
    }
    if (tmp9) {
      xx[1] = atan2(Cartx * tmp7, Carty * sqrt((1.0 / 2.0) * tmp0 + tmp6));
    } else {
      xx[1] = atan2(Cartx * tmp11, Carty * sqrt(2 * tmp0 * tmp10 + tmp0));
    }
    xx[2] = params->SINHWZ * asinh(Cartz * sinh((1.0 / (params->SINHWZ))) / params->AMPLZ);

    // Find the nearest grid indices (i0, i1, i2) for the given Cartesian coordinates (x, y, z).
    // Assuming a cell-centered grid, which follows the pattern:
    //   xx0[i0] = params->xxmin0 + ((REAL)(i0 - NGHOSTS) + 0.5) * params->dxx0
    // The index i0 can be derived as:
    //   i0 = (xx0[i0] - params->xxmin0) / params->dxx0 - 0.5 + NGHOSTS
    // Now, including typecasts:
    //   i0 = (int)((xx[0] - params->xxmin0) / params->dxx0 - 0.5 + (REAL)NGHOSTS)
    // C float-to-int conversion truncates toward zero; for nonnegative inputs this matches floor().
    // Assuming (xx - xxmin)/dxx + NGHOSTS is nonnegative (typical for valid interior points), this is safe.
    //   i0 = (int)((xx[0] - params->xxmin0) / params->dxx0 - 0.5 + (REAL)NGHOSTS + 0.5)
    // The 0.5 values cancel out:
    //   i0 =           (int)( ( xx[0] - params->xxmin0 ) / params->dxx0 + (REAL)NGHOSTS )
    Cart_to_i0i1i2[0] = (int)((xx[0] - params->xxmin0) / params->dxx0 + (REAL)NGHOSTS);
    Cart_to_i0i1i2[1] = (int)((xx[1] - params->xxmin1) / params->dxx1 + (REAL)NGHOSTS);
    Cart_to_i0i1i2[2] = (int)((xx[2] - params->xxmin2) / params->dxx2 + (REAL)NGHOSTS);
  }
} // END FUNCTION: Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTPCylindrical
