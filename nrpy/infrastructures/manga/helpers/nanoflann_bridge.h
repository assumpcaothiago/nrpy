/**
 * C API bridge for nanoflann KD-tree operations used by BHaH.
 *
 * This header is C-compatible and can be included from C and C++ translation units.
 */
#ifndef NANOFLANN_BRIDGE_H
#define NANOFLANN_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle (defined in nanoflann_bridge.cpp).
typedef struct BHaH_KDTree BHaH_KDTree;

/**
 * Build a 3D KD-tree over particle positions stored in a flat array.
 *
 * The particle data is interpreted as:
 * data[p * stride + x_index], data[p * stride + y_index], data[p * stride + z_index].
 *
 * @param npts Number of particles/points.
 * @param data Pointer to the flat particle array (length >= npts * stride).
 * @param stride Number of doubles per particle record.
 * @param x_index Index (within a particle record) of the x coordinate.
 * @param y_index Index (within a particle record) of the y coordinate.
 * @param z_index Index (within a particle record) of the z coordinate.
 * @return Pointer to KD-tree handle, or NULL on failure.
 */
BHaH_KDTree *BHaH_kdtree_build_3d(int npts, const double *data, int stride, int x_index,
                                  int y_index, int z_index);

/**
 * Build a 3D KD-tree over packed xyz positions:
 * xyz[3 * p + 0] = x, xyz[3 * p + 1] = y, xyz[3 * p + 2] = z.
 *
 * @param npts Number of particles/points.
 * @param xyz Pointer to packed xyz array (length >= 3 * npts).
 * @return Pointer to KD-tree handle, or NULL on failure.
 */
BHaH_KDTree *BHaH_kdtree_build_3d_xyz(int npts, const double *xyz);

/**
 * Query the KD-tree for the single nearest neighbor of query_pt (3D).
 *
 * @param tree KD-tree handle returned by build function.
 * @param query_pt Query point (x, y, z).
 * @param out_index Output particle index in [0, npts - 1].
 * @param out_dist2 Output squared Euclidean distance to the nearest neighbor.
 * @return 0 on success; nonzero on failure.
 */
int BHaH_kdtree_query_1nn_3d(const BHaH_KDTree *tree, const double query_pt[3], int *out_index,
                             double *out_dist2);

/**
 * Free a KD-tree handle.
 *
 * @param tree KD-tree handle (may be NULL).
 */
void BHaH_kdtree_free(BHaH_KDTree *tree);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NANOFLANN_BRIDGE_H
