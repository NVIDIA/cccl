//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Structured tensor partitions described as CuTe-style strided layouts
 *
 * A cute_partition describes how a (padded) tensor is distributed over a grid
 * of places as a two-mode layout: a "place" mode enumerating the places and a
 * "local" mode enumerating the elements owned by one place. Both modes are
 * flattened lists of (extent, stride) leaves; strides are in linear element
 * units over the PADDED extents, with dimension 0 varying fastest (the
 * convention of dim4::get_index; row-major front-ends should reverse their
 * dimensions when constructing).
 *
 * Padding is the key soundness ingredient (the CuTe "predication" idiom:
 * partition the rounded-up shape, predicate against the true extents). Each
 * split dimension is padded so the layout is exact and bijective over the
 * padded space, which makes validation O(leaves) and ownership queries a
 * closed-form divmod chain; coordinates beyond the true extents simply own no
 * bytes. No dependency on CUTLASS/CuTe: only the trivial mixed-radix subset
 * of the layout algebra is needed, precisely because exactness is required.
 *
 * This type is a structured *generator* for the owner function consumed by
 * the localized allocation machinery (localized_array,
 * evaluate_localized_placement): it deliberately does not compute placement
 * plans itself - the block-majority engine decides where blocks live.
 *
 * Leaves live in fixed-capacity cuda::std::array storage, so the partition is
 * trivially copyable and its queries (owner(), dims) are host/device callable:
 * it can cross the kernel boundary by value, which a future parallel_for
 * integration relies on.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__places/partitions/blocked_partition.cuh>
#include <cuda/experimental/__places/places.cuh>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cuda::experimental::places
{
/**
 * @brief One (extent, stride) leaf of a flattened layout mode
 *
 * Strides are in linear element units over the padded extents, dimension 0
 * varying fastest.
 */
struct layout_leaf
{
  size_t extent;
  ::std::ptrdiff_t stride;
};

/**
 * @brief Available policies for a tensor dimension in a structured partition
 *
 * Describes how one tensor dimension maps onto the grid: not at all (whole),
 * or distributed over one grid axis with a named policy.
 */
enum class dim_policy
{
  whole, //!< dimension is not distributed
  blocked, //!< contiguous chunks of ceil(extent / places)
  cyclic, //!< round-robin elements
  block_cyclic //!< round-robin blocks of a given size
};

/**
 * @brief Partition specification for one tensor dimension (see dim_policy)
 */
struct dim_spec
{
  dim_policy policy = dim_policy::whole;
  int mesh_axis     = -1; //!< grid axis this dimension distributes over
  size_t block      = 0; //!< block size (block_cyclic only)
};

//! Maximum number of leaves per layout mode (make_partition emits at most 2
//! per dimension; fixed capacity keeps partitions and sub-shapes trivially
//! copyable across the kernel boundary)
inline constexpr size_t cute_partition_max_leaves = 16;

/**
 * @brief The set of element coordinates one place owns, as iterated by
 * parallel_for
 *
 * Produced by cute_partition::apply(): enumerates the place's local mode and
 * converts each local index to global tensor coordinates. Trivially copyable
 * (fixed-capacity leaf storage) so it crosses the kernel boundary by value,
 * and satisfies the shape interface the parallel_for kernels consume
 * (size() + index_to_coords()).
 */
template <size_t rank>
class cute_sub_shape
{
public:
  using coords_t = ::std::array<size_t, rank>;

  /**
   * @param local_leaves Local mode of the place (leaf 0 fastest)
   * @param offset Linear element offset of the place's first element
   * @param padded_dims Padded tensor extents the strides refer to
   * @param lo Inclusive per-dimension lower bounds of the iterated region
   * @param hi Exclusive per-dimension upper bounds of the iterated region;
   *        coordinates outside [lo, hi) are skipped by the parallel_for
   *        loops (predication: interior regions and padding phantoms alike)
   */
  _CCCL_HOST_DEVICE cute_sub_shape(
    ::cuda::std::span<const layout_leaf> local_leaves,
    size_t offset,
    dim4 padded_dims,
    const ::std::array<size_t, rank>& lo,
    const ::std::array<size_t, rank>& hi)
      : num_leaves_(local_leaves.size())
      , offset_(offset)
      , padded_dims_(padded_dims)
      , lo_(lo)
      , hi_(hi)
  {
    for (size_t k = 0; k < num_leaves_; k++)
    {
      leaves_[k] = local_leaves[k];
    }
  }

  //! Whether the given coordinates are within the iterated region
  _CCCL_HOST_DEVICE bool contains(const coords_t& coords) const
  {
    for (size_t d = 0; d < rank; d++)
    {
      if (coords[d] < lo_[d] || coords[d] >= hi_[d])
      {
        return false;
      }
    }
    return true;
  }

  //! Number of elements this place owns
  _CCCL_HOST_DEVICE size_t size() const
  {
    size_t n = 1;
    for (size_t k = 0; k < num_leaves_; k++)
    {
      n *= leaves_[k].extent;
    }
    return n;
  }

  //! Global tensor coordinates of the place's index-th element
  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    // Local index -> linear element index (mixed radix over the local leaves)
    size_t linear = offset_;
    for (size_t k = 0; k < num_leaves_; k++)
    {
      linear += (index % leaves_[k].extent) * static_cast<size_t>(leaves_[k].stride);
      index /= leaves_[k].extent;
    }

    // Linear element index -> coordinates (dimension 0 fastest)
    coords_t coords{};
    for (size_t d = 0; d < rank; d++)
    {
      coords[d] = linear % padded_dims_.get(d);
      linear /= padded_dims_.get(d);
    }
    return coords;
  }

private:
  ::cuda::std::array<layout_leaf, cute_partition_max_leaves> leaves_{};
  size_t num_leaves_ = 0;
  size_t offset_     = 0;
  dim4 padded_dims_;
  ::std::array<size_t, rank> lo_{};
  ::std::array<size_t, rank> hi_{};
};

/**
 * @brief A structured description of a tensor partition over a grid of places
 *
 * See the file-level documentation for the representation. Construct either
 * directly from leaves (expert form) or through make_partition()
 * (per-dimension specification).
 */
class cute_partition
{
public:
  //! Maximum number of leaves per mode (see cute_partition_max_leaves)
  static constexpr size_t max_leaves = cute_partition_max_leaves;

  /**
   * @brief Construct a partition from flattened leaves (expert form)
   *
   * @param place_leaves Leaves of the place mode, leaf 0 fastest; one leaf
   *        per used grid axis (at most max_leaves)
   * @param place_axes Grid axis associated with each place leaf
   * @param local_leaves Leaves of the local mode, leaf 0 fastest (at most
   *        max_leaves)
   * @param padded_dims Padded tensor extents the strides refer to
   * @param true_dims True tensor extents (the predicate)
   * @param grid_dims Extents of the grid of places
   *
   * Throws std::invalid_argument unless the two modes together tile the
   * padded space exactly (bijectivity - validated in O(leaves)).
   */
  cute_partition(const ::std::vector<layout_leaf>& place_leaves,
                 const ::std::vector<int>& place_axes,
                 const ::std::vector<layout_leaf>& local_leaves,
                 dim4 padded_dims,
                 dim4 true_dims,
                 dim4 grid_dims)
      : num_place_leaves_(place_leaves.size())
      , num_local_leaves_(local_leaves.size())
      , padded_dims_(padded_dims)
      , true_dims_(true_dims)
      , grid_dims_(grid_dims)
  {
    if (place_leaves.size() > max_leaves || local_leaves.size() > max_leaves)
    {
      throw ::std::invalid_argument("cute_partition: at most max_leaves leaves are supported per mode");
    }
    if (place_leaves.size() != place_axes.size())
    {
      throw ::std::invalid_argument("cute_partition: one grid axis is required per place leaf");
    }
    ::std::copy(place_leaves.begin(), place_leaves.end(), place_leaves_.begin());
    ::std::copy(place_axes.begin(), place_axes.end(), place_axes_.begin());
    ::std::copy(local_leaves.begin(), local_leaves.end(), local_leaves_.begin());

    validate();

    // Precompute the decode order: all leaves sorted by decreasing stride.
    // For an exact layout, peeling (linear / stride) % extent in this order
    // recovers every leaf coordinate.
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      decode_[num_decode_++] = {place_leaves_[k], /* place leaf index */ static_cast<::std::ptrdiff_t>(k)};
    }
    for (size_t k = 0; k < num_local_leaves_; k++)
    {
      decode_[num_decode_++] = {local_leaves_[k], /* local */ -1};
    }
    ::std::sort(decode_.begin(), decode_.begin() + num_decode_, [](const decode_leaf& a, const decode_leaf& b) {
      return a.leaf.stride > b.leaf.stride;
    });
  }

  //! True tensor extents (the predicate for the padded space)
  _CCCL_HOST_DEVICE const dim4& true_dims() const
  {
    return true_dims_;
  }

  //! Padded tensor extents the leaf strides refer to
  _CCCL_HOST_DEVICE const dim4& padded_dims() const
  {
    return padded_dims_;
  }

  //! Extents of the grid of places
  _CCCL_HOST_DEVICE const dim4& grid_dims() const
  {
    return grid_dims_;
  }

  //! Leaves of the place mode (leaf 0 fastest)
  _CCCL_HOST_DEVICE ::cuda::std::span<const layout_leaf> place_leaves() const
  {
    return {place_leaves_.data(), num_place_leaves_};
  }

  //! Grid axis associated with each place leaf
  _CCCL_HOST_DEVICE ::cuda::std::span<const int> place_axes() const
  {
    return {place_axes_.data(), num_place_leaves_};
  }

  //! Leaves of the local mode (leaf 0 fastest)
  _CCCL_HOST_DEVICE ::cuda::std::span<const layout_leaf> local_leaves() const
  {
    return {local_leaves_.data(), num_local_leaves_};
  }

  //! Number of places the partition distributes over (product of place
  //! extents; grid axes not bound to any dimension receive coordinate 0 and
  //! do not count)
  _CCCL_HOST_DEVICE size_t num_places() const
  {
    size_t p = 1;
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      p *= place_leaves_[k].extent;
    }
    return p;
  }

  //! Number of padded elements owned by each place (product of local extents)
  _CCCL_HOST_DEVICE size_t tiles_per_place() const
  {
    size_t n = 1;
    for (size_t k = 0; k < num_local_leaves_; k++)
    {
      n *= local_leaves_[k].extent;
    }
    return n;
  }

  /**
   * @brief Grid position owning the element at the given coordinates
   *
   * Total on all true coordinates (true extents never exceed the padded
   * ones); grid axes not bound to any dimension get coordinate 0.
   */
  _CCCL_HOST_DEVICE pos4 owner(pos4 data_coords) const
  {
    const size_t linear = padded_dims_.get_index(data_coords);

    ssize_t place_coord[4] = {0, 0, 0, 0};
    for (size_t k = 0; k < num_decode_; k++)
    {
      const decode_leaf& d = decode_[k];
      if (d.leaf.extent <= 1)
      {
        continue;
      }
      const size_t c = (linear / static_cast<size_t>(d.leaf.stride)) % d.leaf.extent;
      if (d.place_leaf >= 0)
      {
        place_coord[static_cast<size_t>(place_axes_[static_cast<size_t>(d.place_leaf)])] = static_cast<ssize_t>(c);
      }
    }

    return pos4(place_coord[0], place_coord[1], place_coord[2], place_coord[3]);
  }

  /**
   * @brief Sub-shape owned by one place, for parallel_for over a grid
   *
   * Follows the partitioner contract of the classic partitioners: the place
   * is given as its linear index in the dispatch loop (pos4(i)), and the
   * returned shape enumerates the coordinates that place owns. Coordinates
   * beyond the true extents (padding phantoms, for uneven covers) are
   * excluded by the sub-shape's predicate rather than by restructuring the
   * iteration (the CuTe predication idiom).
   *
   * @param s Shape of the task (must match the partition's true extents)
   * @param place_position Linear place index in .x (dispatch convention)
   * @param grid_dims Extents of the grid (must match the partition's)
   */
  template <typename S>
  auto apply(const S& s, pos4 place_position, dim4 grid_dims) const
  {
    constexpr size_t rank = S::rank();
    validate_iteration_rank<rank>();

    for (size_t d = 0; d < rank; d++)
    {
      if (static_cast<size_t>(s.extent(d)) != true_dims_.get(d))
      {
        throw ::std::invalid_argument("cute_partition::apply: the task shape does not match the partition's extents");
      }
    }

    ::std::array<size_t, rank> lo{};
    ::std::array<size_t, rank> hi{};
    for (size_t d = 0; d < rank; d++)
    {
      lo[d] = 0;
      hi[d] = true_dims_.get(d);
    }
    return apply_region<rank>(lo, hi, place_position, grid_dims);
  }

  /**
   * @brief Sub-shape owned by one place, restricted to a region of the tensor
   *
   * The box is not a shape: it is a region within the coordinate space of
   * the tensor this partition was built for (the partition remains the
   * authority on the extents). Each place enumerates its own coordinates and
   * the sub-shape's predicate keeps those inside the box - so the iteration
   * chunks stay aligned with data ownership, unlike scale-free partitioners
   * that split the box itself.
   *
   * @param b Region to iterate (must be contained in [0, true extents))
   * @param place_position Linear place index in .x (dispatch convention)
   * @param grid_dims Extents of the grid (must match the partition's)
   */
  template <size_t dims>
  auto apply(const box<dims>& b, pos4 place_position, dim4 grid_dims) const
  {
    validate_iteration_rank<dims>();

    ::std::array<size_t, dims> lo{};
    ::std::array<size_t, dims> hi{};
    for (size_t d = 0; d < dims; d++)
    {
      if (b.get_begin(d) < 0 || static_cast<size_t>(b.get_end(d)) > true_dims_.get(d))
      {
        throw ::std::invalid_argument("cute_partition::apply: the box is not contained in the partition's extents");
      }
      lo[d] = static_cast<size_t>(b.get_begin(d));
      hi[d] = static_cast<size_t>(b.get_end(d));
    }
    return apply_region<dims>(lo, hi, place_position, grid_dims);
  }

  /**
   * @brief Linear element offset (in the padded space) of a place's first
   * element, given the place's linear index in place-mode order (leaf 0
   * fastest)
   */
  size_t place_offset(size_t place_index) const
  {
    if (place_index >= num_places())
    {
      throw ::std::out_of_range("cute_partition::place_offset: place index out of range");
    }
    size_t offset = 0;
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      offset += (place_index % place_leaves_[k].extent) * static_cast<size_t>(place_leaves_[k].stride);
      place_index /= place_leaves_[k].extent;
    }
    return offset;
  }

  //! Structural comparison (used for data place ordering)
  int cmp(const cute_partition& o) const
  {
    const auto cmp_sizes = [](size_t a, size_t b) {
      return (a < b) ? -1 : (a > b) ? 1 : 0;
    };
    const auto cmp_dims = [&cmp_sizes](const dim4& a, const dim4& b) {
      for (size_t axis = 0; axis < 4; axis++)
      {
        if (const int c = cmp_sizes(a.get(axis), b.get(axis)))
        {
          return c;
        }
      }
      return 0;
    };
    if (const int c = cmp_dims(padded_dims_, o.padded_dims_))
    {
      return c;
    }
    if (const int c = cmp_dims(true_dims_, o.true_dims_))
    {
      return c;
    }
    if (const int c = cmp_dims(grid_dims_, o.grid_dims_))
    {
      return c;
    }
    if (const int c = cmp_sizes(num_place_leaves_, o.num_place_leaves_))
    {
      return c;
    }
    if (const int c = cmp_sizes(num_local_leaves_, o.num_local_leaves_))
    {
      return c;
    }
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      if (const int c = cmp_sizes(place_leaves_[k].extent, o.place_leaves_[k].extent))
      {
        return c;
      }
      if (const int c =
            cmp_sizes(static_cast<size_t>(place_leaves_[k].stride), static_cast<size_t>(o.place_leaves_[k].stride)))
      {
        return c;
      }
      if (const int c = cmp_sizes(static_cast<size_t>(place_axes_[k]), static_cast<size_t>(o.place_axes_[k])))
      {
        return c;
      }
    }
    for (size_t k = 0; k < num_local_leaves_; k++)
    {
      if (const int c = cmp_sizes(local_leaves_[k].extent, o.local_leaves_[k].extent))
      {
        return c;
      }
      if (const int c =
            cmp_sizes(static_cast<size_t>(local_leaves_[k].stride), static_cast<size_t>(o.local_leaves_[k].stride)))
      {
        return c;
      }
    }
    return 0;
  }

  bool operator==(const cute_partition& o) const
  {
    return cmp(o) == 0;
  }

  bool operator!=(const cute_partition& o) const
  {
    return !(*this == o);
  }

private:
  template <size_t rank>
  void validate_iteration_rank() const
  {
    static_assert(rank <= 4, "cute_partition supports at most four-dimensional iteration");
    for (size_t d = rank; d < 4; d++)
    {
      if (padded_dims_.get(d) != 1)
      {
        throw ::std::invalid_argument("cute_partition::apply: the iteration rank does not match the partition's "
                                      "extents");
      }
    }
  }

  template <size_t rank>
  cute_sub_shape<rank> apply_region(
    const ::std::array<size_t, rank>& lo,
    const ::std::array<size_t, rank>& hi,
    pos4 place_position,
    dim4 grid_dims) const
  {
    if (!(grid_dims == grid_dims_))
    {
      throw ::std::invalid_argument("cute_partition::apply: the grid does not match the partition's grid extents");
    }

    // The dispatch loop linearizes places into .x
    const pos4 grid_coords = grid_dims_.index_to_pos(static_cast<size_t>(place_position.x));

    size_t offset = 0;
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      offset += static_cast<size_t>(grid_coords.get(static_cast<size_t>(place_axes_[k])))
              * static_cast<size_t>(place_leaves_[k].stride);
    }

    return cute_sub_shape<rank>(local_leaves(), offset, padded_dims_, lo, hi);
  }

  void validate() const
  {
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      const int a = place_axes_[k];
      if (a < 0 || a > 3)
      {
        throw ::std::invalid_argument("cute_partition: place axis out of range");
      }
      if (place_leaves_[k].extent != grid_dims_.get(static_cast<size_t>(a)))
      {
        throw ::std::invalid_argument("cute_partition: place leaf extent does not match its grid axis extent");
      }
      for (size_t j = 0; j < k; j++)
      {
        if (place_axes_[j] == a)
        {
          throw ::std::invalid_argument("cute_partition: grid axis bound to more than one place leaf");
        }
      }
    }

    // Without replication, every grid axis with extent > 1 must be bound to a
    // tensor dimension; otherwise owner() pins that axis to coordinate 0 and
    // the remaining places on that axis own no bytes. Relax this only if
    // replication is introduced.
    if (num_places() != grid_dims_.size())
    {
      throw ::std::invalid_argument(
        "cute_partition: the partition leaves grid places unused (a grid axis with extent > 1 is bound to no "
        "tensor dimension; replication is not supported). Collapse the unused grid axes or bind them to a "
        "tensor dimension.");
    }

    for (size_t d = 0; d < 4; d++)
    {
      if (true_dims_.get(d) < 1 || true_dims_.get(d) > padded_dims_.get(d))
      {
        throw ::std::invalid_argument("cute_partition: true extents must be within [1, padded extents]");
      }
    }

    // Exactness/bijectivity over the padded space: sorted by increasing
    // stride, the leaves must form a mixed radix (each stride equal to the
    // product of the preceding extents) whose total size is the padded size.
    ::cuda::std::array<layout_leaf, 2 * max_leaves> all{};
    size_t num_all = 0;
    for (size_t k = 0; k < num_place_leaves_ + num_local_leaves_; k++)
    {
      const layout_leaf& l = (k < num_place_leaves_) ? place_leaves_[k] : local_leaves_[k - num_place_leaves_];
      if (l.stride < 0)
      {
        throw ::std::invalid_argument("cute_partition: negative strides are not supported");
      }
      if (l.extent == 0)
      {
        throw ::std::invalid_argument("cute_partition: leaf extents must be at least 1");
      }
      if (l.extent > 1)
      {
        all[num_all++] = l;
      }
    }

    ::std::sort(all.begin(), all.begin() + num_all, [](const layout_leaf& a, const layout_leaf& b) {
      return a.stride < b.stride;
    });

    size_t expected_stride = 1;
    for (size_t k = 0; k < num_all; k++)
    {
      if (static_cast<size_t>(all[k].stride) != expected_stride)
      {
        throw ::std::invalid_argument("cute_partition: leaves do not tile the padded space exactly (layout must be "
                                      "exact and bijective)");
      }
      expected_stride *= all[k].extent;
    }
    if (expected_stride != padded_dims_.size())
    {
      throw ::std::invalid_argument("cute_partition: layout size does not match the padded extents");
    }
  }

  struct decode_leaf
  {
    layout_leaf leaf;
    ::std::ptrdiff_t place_leaf; // index into place_leaves_, or -1 for local leaves
  };

  ::cuda::std::array<layout_leaf, max_leaves> place_leaves_{};
  ::cuda::std::array<int, max_leaves> place_axes_{};
  ::cuda::std::array<layout_leaf, max_leaves> local_leaves_{};
  ::cuda::std::array<decode_leaf, 2 * max_leaves> decode_{};
  size_t num_place_leaves_ = 0;
  size_t num_local_leaves_ = 0;
  size_t num_decode_       = 0;
  dim4 padded_dims_;
  dim4 true_dims_;
  dim4 grid_dims_;
};

/**
 * @brief Build a partition from a per-dimension specification
 *
 * Each entry of `spec` describes how the corresponding tensor dimension maps
 * onto the grid ("blocked over axis 0", ...). Split dimensions are padded up
 * to divisibility, which is what makes the resulting layout exact (see the
 * file-level documentation). Every grid axis with extent > 1 must be bound by
 * some entry; unbound axes would leave those places idle (replication is not
 * supported) and are rejected at construction time.
 *
 * @param true_dims True tensor extents (dimension 0 fastest)
 * @param spec One entry per tensor dimension (at most 4)
 * @param grid_dims Extents of the grid of places
 */
inline cute_partition make_partition(dim4 true_dims, const ::std::vector<dim_spec>& spec, dim4 grid_dims)
{
  if (spec.size() > 4)
  {
    throw ::std::invalid_argument("make_partition: at most 4 dimensions are supported");
  }
  const size_t rank = spec.size();

  // Pass 1: padded extent per dimension
  ::std::array<size_t, 4> padded = {1, 1, 1, 1};
  for (size_t d = 0; d < 4; d++)
  {
    const size_t extent = true_dims.get(d);
    if (d >= rank || spec[d].policy == dim_policy::whole)
    {
      padded[d] = extent;
      continue;
    }

    const auto& e = spec[d];
    if (e.mesh_axis < 0 || e.mesh_axis > 3)
    {
      throw ::std::invalid_argument("make_partition: mesh_axis out of range");
    }
    const size_t nplaces = grid_dims.get(static_cast<size_t>(e.mesh_axis));
    if (nplaces == 0)
    {
      throw ::std::invalid_argument("make_partition: grid axis extents must be at least 1");
    }

    switch (e.policy)
    {
      case dim_policy::blocked:
      case dim_policy::cyclic: {
        const size_t chunk = (extent + nplaces - 1) / nplaces;
        padded[d]          = chunk * nplaces;
        break;
      }
      case dim_policy::block_cyclic: {
        if (e.block == 0)
        {
          throw ::std::invalid_argument("make_partition: block_cyclic requires a block size");
        }
        const size_t super  = e.block * nplaces;
        const size_t nsuper = (extent + super - 1) / super;
        padded[d]           = nsuper * super;
        break;
      }
      default:
        break;
    }
  }

  // Pass 2: dimension strides over the padded extents (dimension 0 fastest)
  ::std::array<size_t, 4> stride = {1, 1, 1, 1};
  for (size_t d = 1; d < 4; d++)
  {
    stride[d] = stride[d - 1] * padded[d - 1];
  }

  // Pass 3: leaves, fastest dimension first
  ::std::vector<layout_leaf> place_leaves;
  ::std::vector<int> place_axes;
  ::std::vector<layout_leaf> local_leaves;

  for (size_t d = 0; d < 4; d++)
  {
    const size_t R = stride[d];
    if (d >= rank || spec[d].policy == dim_policy::whole)
    {
      if (padded[d] > 1)
      {
        local_leaves.push_back({padded[d], static_cast<::std::ptrdiff_t>(R)});
      }
      continue;
    }

    const auto& e        = spec[d];
    const size_t nplaces = grid_dims.get(static_cast<size_t>(e.mesh_axis));

    switch (e.policy)
    {
      case dim_policy::blocked: {
        const size_t b = padded[d] / nplaces;
        local_leaves.push_back({b, static_cast<::std::ptrdiff_t>(R)});
        place_leaves.push_back({nplaces, static_cast<::std::ptrdiff_t>(b * R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      case dim_policy::cyclic: {
        local_leaves.push_back({padded[d] / nplaces, static_cast<::std::ptrdiff_t>(nplaces * R)});
        place_leaves.push_back({nplaces, static_cast<::std::ptrdiff_t>(R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      case dim_policy::block_cyclic: {
        const size_t nsuper = padded[d] / (e.block * nplaces);
        local_leaves.push_back({e.block, static_cast<::std::ptrdiff_t>(R)});
        local_leaves.push_back({nsuper, static_cast<::std::ptrdiff_t>(e.block * nplaces * R)});
        place_leaves.push_back({nplaces, static_cast<::std::ptrdiff_t>(e.block * R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      default:
        break;
    }
  }

  return cute_partition(
    mv(place_leaves),
    mv(place_axes),
    mv(local_leaves),
    dim4(padded[0], padded[1], padded[2], padded[3]),
    true_dims,
    grid_dims);
}

/**
 * @brief Evaluate - without allocating - how a localized allocation of a
 * tensor distributed by `partition` over `grid` would be placed
 *
 * See evaluate_localized_placement(); the tensor extents are the partition's
 * true extents.
 */
[[nodiscard]] inline localized_stats evaluate_localized_placement(
  const exec_place& grid,
  const cute_partition& partition,
  size_t elemsize,
  size_t probes     = localized_placement_default_probes,
  size_t block_size = 0)
{
  if (!(grid.get_dims() == partition.grid_dims()))
  {
    throw ::std::invalid_argument("the partition's grid extents do not match the execution place grid");
  }

  const dim4 data_dims = partition.true_dims();

  if (block_size == 0)
  {
    block_size = default_placement_block_size();
  }

  localized_stats stats;

  const size_t total_elems = data_dims.size();
  stats.total_bytes        = total_elems * elemsize;
  stats.vm_bytes           = ((stats.total_bytes + block_size - 1) / block_size) * block_size;
  stats.block_size         = block_size;
  stats.nblocks            = stats.vm_bytes / block_size;

  const ::std::vector<pos4> owners = compute_block_owners(
    [&](size_t index) {
      return partition.owner(data_dims.index_to_pos(index));
    },
    stats.nblocks,
    block_size,
    elemsize,
    total_elems,
    probes,
    stats);

  for_each_owner_run(owners, [&](pos4 p, size_t /*first_block*/, size_t num_blocks) {
    const data_place place = grid.get_place(p).affine_data_place();
    stats.bytes_per_place[place.to_string()] += num_blocks * block_size;
    stats.bytes_per_grid_index[grid.get_dims().get_index(p)] += num_blocks * block_size;
    stats.nallocs++;
  });

  return stats;
}

/**
 * @brief Composite data place whose partitioner is a cute_partition object
 *
 * Like data_place_composite but the owner function is stateful (a bare
 * partition_fn_t cannot carry the partition's leaves), so the partition
 * object is stored on the place. Because a padded partition is intrinsically
 * specific to one tensor, such a place is per-tensor by nature; the reusable
 * shape-free policy object remains the partition_fn_t composite. This place
 * supports both shaped raw allocations (allocate_nd(data_dims, elemsize)) and
 * STF logical data.
 */
class data_place_cute_composite final : public data_place_interface
{
public:
  data_place_cute_composite(exec_place grid, cute_partition partition)
      : grid_(mv(grid))
      , partition_(mv(partition))
  {
    if (!(grid_.get_dims() == partition_.grid_dims()))
    {
      throw ::std::invalid_argument("the partition's grid extents do not match the execution place grid");
    }
  }

  bool is_resolved() const override
  {
    return true;
  }

  bool is_composite() const override
  {
    return true;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::composite;
  }

  ::std::string to_string() const override
  {
    return "composite_cute";
  }

  size_t hash() const override
  {
    throw ::std::logic_error("hash() not supported for composite data_place");
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    const auto& o = static_cast<const data_place_cute_composite&>(other);
    if (int c = partition_.cmp(o.partition_))
    {
      return c;
    }
    if (grid_ == o.grid_)
    {
      return 0;
    }
    return (grid_ < o.grid_) ? -1 : 1;
  }

  void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    throw ::std::runtime_error(
      "cute-partition composite data_place cannot allocate from a byte count alone: use allocate_nd with the "
      "partition's true extents or allocate through logical data");
  }

  void* allocate_nd(dim4 data_dims, size_t elemsize, cudaStream_t) const override
  {
    // A padded partition is specific to one tensor: the requested extents
    // must be the ones the partition was built for.
    if (!(data_dims == partition_.true_dims()))
    {
      throw ::std::invalid_argument("cute composite data_place: requested extents do not match the partition's true "
                                    "extents");
    }

    auto arr = ::std::make_unique<localized_array>(
      grid_,
      ::std::function<pos4(size_t)>([this, data_dims](size_t index) {
        return partition_.owner(data_dims.index_to_pos(index));
      }),
      data_dims.size(),
      elemsize,
      data_dims);
    void* ptr                           = arr->get_base_ptr();
    get_composite_alloc_registry()[ptr] = ::std::move(arr);
    return ptr;
  }

  void deallocate(void* ptr, size_t, cudaStream_t) const override
  {
    deallocate_composite_data_place(ptr);
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }

  ::std::shared_ptr<void> get_affine_exec_impl() const override
  {
    return grid_.get_impl();
  }

  const cute_partition& get_partition() const
  {
    return partition_;
  }

  const exec_place& get_grid() const
  {
    return grid_;
  }

private:
  exec_place grid_;
  cute_partition partition_;
};

/**
 * @brief Create a composite data place backed by a cute_partition
 */
inline data_place make_composite_data_place(const exec_place& grid, cute_partition partition)
{
  return data_place(::std::make_shared<data_place_cute_composite>(grid, mv(partition)));
}

#ifdef UNITTESTED_FILE
UNITTEST("make_partition blocked leaves and owners")
{
  // 2-D tensor (6, 4), dimension 1 blocked over 2 places (axis 0)
  const dim4 true_dims(6, 4);
  const dim4 grid_dims(2);
  auto part = make_partition(true_dims, {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, grid_dims);

  EXPECT(part.padded_dims() == dim4(6, 4));
  EXPECT(part.num_places() == 2);
  EXPECT(part.place_offset(0) == 0);
  EXPECT(part.place_offset(1) == 12); // 2 rows of 6

  for (size_t y = 0; y < 4; y++)
  {
    for (size_t x = 0; x < 6; x++)
    {
      EXPECT(part.owner(pos4(x, y)) == pos4(y / 2));
    }
  }
};

UNITTEST("make_partition pads uneven blocked dimensions")
{
  // (4, 5) tensor blocked over 2 places along dimension 1: chunk = 3, so the
  // padded extent is 6. This is the aliasing regression: without padding, an
  // unclamped layout would leak coordinates of one place into another.
  const dim4 true_dims(4, 5);
  const dim4 grid_dims(2);
  auto part = make_partition(true_dims, {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, grid_dims);

  EXPECT(part.padded_dims() == dim4(4, 6));

  size_t counts[2] = {0, 0};
  for (size_t y = 0; y < 5; y++)
  {
    for (size_t x = 0; x < 4; x++)
    {
      const pos4 o = part.owner(pos4(x, y));
      EXPECT(o == pos4(y / 3));
      counts[static_cast<size_t>(o.x)]++;
    }
  }
  // Place 0 owns columns 0-2, place 1 owns columns 3-4 of the true extents
  EXPECT(counts[0] == 4 * 3);
  EXPECT(counts[1] == 4 * 2);
};

UNITTEST("make_partition cyclic and block_cyclic owners")
{
  const dim4 grid_dims(2);

  auto cyc = make_partition(dim4(7), {dim_spec{dim_policy::cyclic, 0, 0}}, grid_dims);
  for (size_t x = 0; x < 7; x++)
  {
    EXPECT(cyc.owner(pos4(x)) == pos4(x % 2));
  }

  auto bc = make_partition(dim4(8), {dim_spec{dim_policy::block_cyclic, 0, 2}}, grid_dims);
  for (size_t x = 0; x < 8; x++)
  {
    EXPECT(bc.owner(pos4(x)) == pos4((x / 2) % 2));
  }
};

UNITTEST("cute_partition owner matches blocked_partition get_executor")
{
  // Same policy expressed via make_partition and via the classic partitioner
  const dim4 true_dims(10);
  const dim4 grid_dims(3);
  auto part = make_partition(true_dims, {dim_spec{dim_policy::blocked, 0, 0}}, grid_dims);

  for (size_t x = 0; x < 10; x++)
  {
    pos4 expected;
    blocked_partition_custom<0>::get_executor(&expected, pos4(x), true_dims, grid_dims);
    EXPECT(part.owner(pos4(x)) == expected);
  }
};

UNITTEST("cute_partition comparison includes complete dimensions")
{
  const dim4 true_dims(2, 4);
  const dim4 grid_dims(2);

  // These layouts have the same total padded size and identical leaves, but
  // their multidimensional strides map coordinates to different owners.
  const cute_partition a({{2, 1}}, {0}, {{6, 2}}, dim4(2, 6), true_dims, grid_dims);
  const cute_partition b({{2, 1}}, {0}, {{6, 2}}, dim4(3, 4), true_dims, grid_dims);

  EXPECT(a != b);
  EXPECT(!(a.owner(pos4(0, 1)) == b.owner(pos4(0, 1))));
};

UNITTEST("cute_partition rejects lower-rank iteration")
{
  // Dimension 1 has a true extent of one but is padded to two by its split;
  // omitting it would still discard ownership coordinates and duplicate work.
  const dim4 true_dims(8, 1);
  const dim4 grid_dims(2, 2);
  const auto part =
    make_partition(true_dims, {dim_spec{dim_policy::blocked, 0, 0}, dim_spec{dim_policy::blocked, 1, 0}}, grid_dims);
  const box<1> line({0ul, 8ul});

  bool thrown = false;
  try
  {
    part.apply(line, pos4(0), grid_dims);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);
};

UNITTEST("cute_partition validation rejects inexact layouts")
{
  const dim4 dims(8);
  const dim4 grid(2);

  // Overlapping: both leaves have stride 1
  bool thrown = false;
  try
  {
    cute_partition({{2, 1}}, {0}, {{4, 1}}, dims, dims, grid);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);

  // Under-covering: strides tile only half of the padded space
  thrown = false;
  try
  {
    cute_partition({{2, 4}}, {0}, {{2, 1}}, dims, dims, grid);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);
};

UNITTEST("make_partition rejects partitions that leave grid places unused")
{
  const dim4 true_dims(64);
  const dim4 grid_dims(6, 4);

  // Blocked on axis 0 only: 6 of 24 places would own data; axes 1..3 idle.
  bool thrown = false;
  try
  {
    make_partition(true_dims, {dim_spec{dim_policy::blocked, 0, 0}}, grid_dims);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);

  // Binding every grid axis uses all places.
  auto part =
    make_partition(dim4(12, 8), {dim_spec{dim_policy::blocked, 0, 0}, dim_spec{dim_policy::blocked, 1, 0}}, grid_dims);
  EXPECT(part.num_places() == grid_dims.size());
};
#endif // UNITTESTED_FILE
} // namespace cuda::experimental::places
