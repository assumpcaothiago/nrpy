// nanoflann_bridge.cpp
//
// C++ implementation of a C ABI bridge around nanoflann.
// This lets pure-C BHaH code build/query a KD-tree without seeing C++ templates.

#include "nanoflann_bridge.h"

// nanoflann header-only library:
#include "nanoflann.hpp"

#include <new>     // std::nothrow
#include <cstddef> // std::size_t

namespace {

/* Adaptor over a flat array: data[p*stride + coord_index]. */
struct FlatArrayPointCloudAdaptor {
  const double *data{nullptr};
  int npts{0};
  int stride{0};
  int x_index{0};
  int y_index{0};
  int z_index{0};

  FlatArrayPointCloudAdaptor(const double *data_, int npts_, int stride_, int xi, int yi, int zi)
      : data(data_), npts(npts_), stride(stride_), x_index(xi), y_index(yi), z_index(zi) {}

  inline std::size_t kdtree_get_point_count() const { return static_cast<std::size_t>(npts); }

  inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
    const std::size_t base = idx * static_cast<std::size_t>(stride);
    if (dim == 0)
      return data[base + static_cast<std::size_t>(x_index)];
    if (dim == 1)
      return data[base + static_cast<std::size_t>(y_index)];
    return data[base + static_cast<std::size_t>(z_index)];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX &) const {
    return false; // no bounding-box acceleration
  }
};

using Metric = nanoflann::L2_Simple_Adaptor<double, FlatArrayPointCloudAdaptor>;
using Index = nanoflann::KDTreeSingleIndexAdaptor<Metric, FlatArrayPointCloudAdaptor, 3, std::size_t>;

} // namespace

/* Definition of the opaque handle. */
struct BHaH_KDTree {
  FlatArrayPointCloudAdaptor adaptor;
  Index index;

  BHaH_KDTree(const double *data, int npts, int stride, int xi, int yi, int zi, int leaf_max_size)
      : adaptor(data, npts, stride, xi, yi, zi),
        index(3 /*dim*/, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size)) {}
};

namespace {

static BHaH_KDTree *BHaH_kdtree_build_3d_impl(int npts, const double *data, int stride,
                                              int x_index, int y_index, int z_index) {
  // Basic validation
  if (npts <= 0 || data == nullptr || stride <= 0)
    return nullptr;
  if (x_index < 0 || y_index < 0 || z_index < 0)
    return nullptr;
  if (x_index >= stride || y_index >= stride || z_index >= stride)
    return nullptr;

  // Tunable: leaf size controls build/query tradeoff.
  const int leaf_max_size = 10;

  try {
    BHaH_KDTree *tree =
        new (std::nothrow) BHaH_KDTree(data, npts, stride, x_index, y_index, z_index, leaf_max_size);
    if (!tree)
      return nullptr;

    tree->index.buildIndex();
    return tree;
  } catch (...) {
    // Never let exceptions cross a C ABI boundary.
    return nullptr;
  }
}

} // namespace

extern "C" BHaH_KDTree *BHaH_kdtree_build_3d(int npts, const double *data, int stride,
                                             int x_index, int y_index, int z_index) {
  return BHaH_kdtree_build_3d_impl(npts, data, stride, x_index, y_index, z_index);
}

extern "C" BHaH_KDTree *BHaH_kdtree_build_3d_xyz(int npts, const double *xyz) {
  // xyz is packed: stride=3, indices 0,1,2
  return BHaH_kdtree_build_3d_impl(npts, xyz, 3, 0, 1, 2);
}

extern "C" int BHaH_kdtree_query_1nn_3d(const BHaH_KDTree *tree, const double query_pt[3],
                                        int *out_index, double *out_dist2) {
  if (tree == nullptr || query_pt == nullptr || out_index == nullptr || out_dist2 == nullptr)
    return 1;

  try {
    std::size_t ret_index = 0;
    double ret_dist2 = 0.0;

    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&ret_index, &ret_dist2);

    nanoflann::SearchParameters params;
    tree->index.findNeighbors(resultSet, query_pt, params);

    *out_index = static_cast<int>(ret_index);
    *out_dist2 = ret_dist2;
    return 0;
  } catch (...) {
    return 2;
  }
}

extern "C" void BHaH_kdtree_free(BHaH_KDTree *tree) {
  delete tree;
}
