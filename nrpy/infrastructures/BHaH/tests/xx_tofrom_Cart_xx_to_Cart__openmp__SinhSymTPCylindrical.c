#include "BHaH_defines.h"

/**
 * Compute Cartesian coordinates {x, y, z} = {xCart[0], xCart[1], xCart[2]} from the
 * local coordinate vector {xx[0], xx[1], xx[2]} = {xx0, xx1, xx2},
 * taking into account the possibility that the origin of this grid is off-center.
 */
void xx_to_Cart__rfm__SinhSymTPCylindrical(const params_struct *restrict params, const REAL xx[3], REAL xCart[3]) {
  const REAL xx0 = xx[0];
  const REAL xx1 = xx[1];
  const REAL xx2 = xx[2];
  /*
   *  Original SymPy expressions:
   *  "[xCart[0] = params->Cart_originx + sqrt(params->AMPLXY**2*(exp(xx0/params->SINHWXY) - exp(-xx0/params->SINHWXY))**2/(exp(1/params->SINHWXY) -
   * exp(-1/params->SINHWXY))**2 + params->bScaleXY**2)*sin(xx1)]"
   *  "[xCart[1] = params->AMPLXY*(exp(xx0/params->SINHWXY) - exp(-xx0/params->SINHWXY))*cos(xx1)/(exp(1/params->SINHWXY) - exp(-1/params->SINHWXY)) +
   * params->Cart_originy]"
   *  "[xCart[2] = params->AMPLZ*(exp(xx2/params->SINHWZ) - exp(-xx2/params->SINHWZ))/(exp(1/params->SINHWZ) - exp(-1/params->SINHWZ)) +
   * params->Cart_originz]"
   */
  {
    const REAL tmp0 = (1.0 / (params->SINHWXY));
    const REAL tmp4 = (1.0 / (params->SINHWZ));
    const REAL tmp1 = exp(tmp0) - exp(-tmp0);
    const REAL tmp3 = exp(tmp0 * xx0) - exp(-tmp0 * xx0);
    xCart[0] =
        params->Cart_originx +
        sqrt(((params->AMPLXY) * (params->AMPLXY)) * ((tmp3) * (tmp3)) / ((tmp1) * (tmp1)) + ((params->bScaleXY) * (params->bScaleXY))) * sin(xx1);
    xCart[1] = params->AMPLXY * tmp3 * cos(xx1) / tmp1 + params->Cart_originy;
    xCart[2] = params->AMPLZ * (exp(tmp4 * xx2) - exp(-tmp4 * xx2)) / (exp(tmp4) - exp(-tmp4)) + params->Cart_originz;
  }
} // END FUNCTION: xx_to_Cart__rfm__SinhSymTPCylindrical
