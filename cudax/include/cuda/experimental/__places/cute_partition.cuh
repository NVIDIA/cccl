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
 * Typed partitions keep rank and leaf counts in their type and store only
 * their exact leaves. This preserves compile-time loop bounds for
 * parallel_for, while a canonical fixed-capacity descriptor provides runtime
 * type erasure for data places and C/Python interoperability.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/optional>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__tuple_dir/apply.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>
#include <cuda/std/tuple>

#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__places/partitions/blocked_partition.cuh>
#include <cuda/experimental/__places/places.cuh>

#include <algorithm>
#include <optional>
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
  ::cuda::std::ptrdiff_t stride;
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
 * Produced by cute_partition_descriptor::apply(): enumerates the place's
 * local mode and converts each local index to global tensor coordinates. This
 * runtime fallback uses fixed-capacity storage; typed C++ partitions produce
 * static_cute_sub_shape instead.
 */
template <size_t rank>
class cute_sub_shape
{
public:
  using coords_t = ::cuda::std::array<size_t, rank>;

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
    const ::cuda::std::array<size_t, rank>& lo,
    const ::cuda::std::array<size_t, rank>& hi)
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
  ::cuda::std::array<size_t, rank> lo_{};
  ::cuda::std::array<size_t, rank> hi_{};
};

/**
 * @brief Compact sub-shape with a compile-time number of layout leaves
 *
 * This is the kernel-facing representation produced by a typed
 * cute_partition. Unlike cute_sub_shape, it stores exactly the leaves used by
 * the partition and has no runtime leaf count, so decoding can be unrolled.
 */
template <size_t rank, size_t num_leaves>
class static_cute_sub_shape
{
public:
  using coords_t = ::cuda::std::array<size_t, rank>;

  _CCCL_HOST_DEVICE static_cute_sub_shape(
    const ::cuda::std::array<layout_leaf, num_leaves>& local_leaves,
    size_t offset,
    dim4 padded_dims,
    const ::cuda::std::array<size_t, rank>& lo,
    const ::cuda::std::array<size_t, rank>& hi)
      : leaves_(local_leaves)
      , offset_(offset)
      , padded_dims_(padded_dims)
      , lo_(lo)
      , hi_(hi)
  {}

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

  _CCCL_HOST_DEVICE size_t size() const
  {
    size_t n = 1;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t k = 0; k < num_leaves; k++)
    {
      n *= leaves_[k].extent;
    }
    return n;
  }

  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    size_t linear = offset_;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t k = 0; k < num_leaves; k++)
    {
      linear += (index % leaves_[k].extent) * static_cast<size_t>(leaves_[k].stride);
      index /= leaves_[k].extent;
    }

    coords_t coords{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t d = 0; d < rank; d++)
    {
      coords[d] = linear % padded_dims_.get(d);
      linear /= padded_dims_.get(d);
    }
    return coords;
  }

private:
  ::cuda::std::array<layout_leaf, num_leaves> leaves_{};
  size_t offset_ = 0;
  dim4 padded_dims_;
  ::cuda::std::array<size_t, rank> lo_{};
  ::cuda::std::array<size_t, rank> hi_{};
};

/**
 * @brief Canonical runtime description of a structured tensor partition
 *
 * Used at polymorphic and ABI boundaries where one concrete C++ partition
 * type cannot be retained. Normal C++ code should construct a typed
 * cute_partition through make_partition().
 */
class cute_partition_descriptor
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
  cute_partition_descriptor(
    const ::std::vector<layout_leaf>& place_leaves,
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
      _CCCL_THROW(::std::invalid_argument, "cute_partition: at most max_leaves leaves are supported per mode");
    }
    if (place_leaves.size() != place_axes.size())
    {
      _CCCL_THROW(::std::invalid_argument, "cute_partition: one grid axis is required per place leaf");
    }
    ::cuda::std::copy(place_leaves.begin(), place_leaves.end(), place_leaves_.begin());
    ::cuda::std::copy(place_axes.begin(), place_axes.end(), place_axes_.begin());
    ::cuda::std::copy(local_leaves.begin(), local_leaves.end(), local_leaves_.begin());

    validate();

    // Precompute the decode order: all leaves sorted by decreasing stride.
    // For an exact layout, peeling (linear / stride) % extent in this order
    // recovers every leaf coordinate.
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      decode_[num_decode_++] = {place_leaves_[k], /* place leaf index */
                                static_cast<::cuda::std::ptrdiff_t>(k)};
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

  /**
   * @brief Enumerate the certified constant-owner byte runs of the flat
   * allocation, merged across adjacent equal-owner segments, in increasing
   * byte order: calls `emit(byte_start, byte_len, owner)` per run; emit
   * returns false to abort the walk early (the enumeration then also
   * returns false).
   *
   * Returns false (without emitting) when the walk would exceed max_runs;
   * see try_block_owners for the budget semantics. The certificate for
   * one owner() evaluation per segment is the divisibility chain of the
   * exact tiling's sorted leaf strides (owner constant on intervals of the
   * padded space aligned to the smallest place-leaf stride).
   */
  template <typename F>
  bool for_each_owner_byte_run(size_t elemsize, size_t max_runs, F&& emit) const
  {
    const size_t t0 = true_dims_.get(0), t1 = true_dims_.get(1);
    const size_t t2 = true_dims_.get(2), t3 = true_dims_.get(3);

    const size_t s = min_owner_run_elems();
    if (s == 0)
    {
      // no place mode, or every place leaf has extent <= 1: single owner
      return emit(size_t(0), t0 * t1 * t2 * t3 * elemsize, owner(pos4(0, 0, 0, 0)));
    }
    const size_t nrows = t1 * t2 * t3;
    if (nrows * (t0 / s + 2) > max_runs)
    {
      return false;
    }

    const size_t ps1 = padded_dims_.get(0);
    const size_t ps2 = ps1 * padded_dims_.get(1);
    const size_t ps3 = ps2 * padded_dims_.get(2);

    // pending merged run
    size_t run_start = 0, run_len = 0;
    pos4 run_owner;
    bool keep_going = true;
    auto push       = [&](size_t start, size_t len, pos4 o) {
      if (run_len > 0 && run_owner == o && run_start + run_len == start)
      {
        run_len += len;
        return;
      }
      if (run_len > 0)
      {
        keep_going = emit(run_start, run_len, run_owner);
      }
      run_start = start;
      run_len   = len;
      run_owner = o;
    };

    size_t ind = 0; // linear element index in the allocation
    for (size_t x3 = 0; keep_going && x3 < t3; x3++)
    {
      for (size_t x2 = 0; keep_going && x2 < t2; x2++)
      {
        for (size_t x1 = 0; keep_going && x1 < t1; x1++)
        {
          const size_t row_pad_base = x1 * ps1 + x2 * ps2 + x3 * ps3;
          size_t x0                 = 0;
          while (keep_going && x0 < t0)
          {
            const size_t pad_pos = row_pad_base + x0;
            const size_t seg     = ::cuda::std::min(t0 - x0, s - (pad_pos % s));
            push((ind + x0) * elemsize, seg * elemsize, owner(pos4(x0, x1, x2, x3))); // pos4 widens to ssize_t; dims
                                                                                      // may exceed INT_MAX
            x0 += seg;
          }
          ind += t0;
        }
      }
    }
    if (keep_going && run_len > 0)
    {
      keep_going = emit(run_start, run_len, run_owner);
    }
    return keep_going;
  }

  /**
   * @brief Certified owner-run granularity in bytes: the smallest place-leaf
   * stride times the element size.
   *
   * Ownership is constant on padded-linear intervals of this size (see
   * try_block_owners), so it is the "smallest part" of the placement layout:
   *  - >= the placement-block size: blocks straddle at most one ownership
   *    boundary and the analytic plan is exact or near-exact;
   *  - <  the placement-block size: NO block can be provably pure -- the
   *    plan is all-straddles with closed-form misplacement, a signal that
   *    identity storage is block-hostile for this spec (prefer a relayout);
   *  - the analytic walk costs ~total_bytes / this many owner() calls,
   *    which is what try_block_owners' max_runs guards.
   *
   * Returns 0 when there is no place mode (single-owner layouts).
   */
  size_t min_owner_run_bytes(size_t elemsize) const
  {
    return min_owner_run_elems() * elemsize;
  }

  //! The same granularity in ELEMENTS of the padded space (the walk's
  //! native unit); 0 means no ownership boundary at all (single owner).
  size_t min_owner_run_elems() const
  {
    // Leaves with extent <= 1 never change the owning coordinate (owner()
    // skips them); including them would drag the minimum down -- extent-1
    // leaves from unit grid axes carry stride 1 (or 0 from the expert
    // constructor) and would falsely collapse the granularity to the element
    // pitch (or trip a division by zero downstream).
    size_t s = 0;
    for (size_t k = 0; k < num_place_leaves_; k++)
    {
      if (place_leaves_[k].extent > 1)
      {
        const auto stride = static_cast<size_t>(place_leaves_[k].stride);
        s                 = (s == 0) ? stride : ::cuda::std::min(s, stride);
      }
    }
    return s;
  }

  /**
   * @brief Analytic per-placement-block owners: divide the ownership layout
   * by the placement-block layout, without sampling.
   *
   * Owners are derived with one owner() evaluation per provably-constant
   * run. The certificate comes from the leaf algebra: in an exact tiling the
   * sorted leaf strides form a divisibility chain, so every place-mode
   * coordinate is constant on intervals of the padded linear space aligned
   * to the smallest place-leaf stride. Blocks straddling an ownership
   * boundary are assigned by exact byte majority and their error is
   * accumulated in *misplaced_bytes (0 means the partition factors through
   * placement blocks and the plan is exact).
   *
   * The linearization convention matches the composite allocation path
   * (dimension 0 varies fastest, see dim4::index_to_pos).
   *
   * @param max_runs evaluation budget for the walk; 0 (the default)
   *        self-scales to max(16 x nblocks, 1 << 16) -- a small constant of
   *        owner() evaluations per placement block produced, so the budget
   *        grows with the allocation instead of being an arbitrary constant.
   *
   * @return per-block owners, or nullopt when the run enumeration would
   *         exceed max_runs (dense sub-block interleavings such as
   *         element-cyclic): callers fall back to sampled majority.
   */
  ::cuda::std::optional<::std::vector<pos4>>
  try_block_owners(size_t block_size_bytes, size_t elemsize, size_t* misplaced_bytes, size_t max_runs = 0) const
  {
    _CCCL_ASSERT(elemsize > 0 && block_size_bytes >= elemsize, "invalid block geometry");
    // the allocation always covers exactly the partition's true extents
    // (composite allocation asserts this before reaching us)
    const size_t total_elems = true_dims_.size();
    const size_t total_bytes = total_elems * elemsize;
    const size_t nblocks     = (total_bytes + block_size_bytes - 1) / block_size_bytes;
    if (max_runs == 0)
    {
      // Auto budget, proportional to the OUTPUT size rather than a fixed
      // constant: 16 owner() evaluations per placement block produced keeps
      // the walk within a small factor of what the sampled fallback would
      // spend on the same allocation (localized_placement_default_probes per
      // block), so accepting the analytic path is never a meaningful cost
      // regression at any allocation size, while dense layouts (runs at
      // element pitch, i.e. thousands of runs per block) are declined
      // immediately. The floor keeps small allocations permissive: below it
      // the walk costs microseconds either way, and declining fine-grained
      // small cases would forfeit exact plans for no measurable saving.
      max_runs = ::cuda::std::max<size_t>(16 * nblocks, size_t(1) << 16);
    }
    if (misplaced_bytes)
    {
      *misplaced_bytes = 0;
    }

    ::std::vector<pos4> owners;
    owners.reserve(nblocks);

    // Streaming block census over the certified runs.
    ::std::vector<::std::pair<pos4, size_t>> hist; // bytes per owner, current block
    size_t cur_block = 0;

    auto close_block = [&]() {
      pos4 best;
      size_t best_bytes = 0, sum = 0;
      for (const auto& e : hist)
      {
        sum += e.second;
        if (e.second > best_bytes)
        {
          best_bytes = e.second;
          best       = e.first;
        }
      }
      _CCCL_ASSERT(!hist.empty(), "empty placement block census");
      owners.push_back(best);
      if (misplaced_bytes)
      {
        *misplaced_bytes += sum - best_bytes;
      }
      hist.clear();
    };

    auto feed = [&](size_t byte_start, size_t byte_len, pos4 o) -> bool {
      while (byte_len > 0)
      {
        const size_t block_end = (cur_block + 1) * block_size_bytes;
        if (byte_start >= block_end)
        {
          close_block();
          cur_block++;
          continue;
        }
        const size_t chunk = ::cuda::std::min(byte_len, block_end - byte_start);
        bool found         = false;
        for (auto& e : hist)
        {
          if (e.first == o)
          {
            e.second += chunk;
            found = true;
            break;
          }
        }
        if (!found)
        {
          hist.emplace_back(o, chunk);
        }
        byte_start += chunk;
        byte_len -= chunk;
      }
      return true;
    };

    if (!for_each_owner_byte_run(elemsize, max_runs, feed))
    {
      return ::cuda::std::nullopt;
    }
    if (!hist.empty())
    {
      close_block();
    }
    _CCCL_ASSERT(owners.size() == nblocks, "placement block census incomplete");
    return owners;
  }

  /**
   * @brief Divide the ownership layout by the placement-block layout,
   * returning the quotient DIRECTLY as maximal same-owner block runs.
   *
   * This is the exact tier's natural output: one run per (cuMemCreate,
   * cuMemMap) pair, with no per-block materialization and no downstream
   * merge -- the census/merge round-trip of the one-owner-per-block
   * representation exists only for tiers whose primitive is per-block
   * (sampling, straddle majority). Succeeds iff every internal ownership
   * boundary is aligned to the block size (the strict quotient exists, so
   * the plan is exact and misplacement is zero by construction); returns
   * nullopt otherwise -- callers fall through to try_block_owners (census
   * with closed-form majority) and then to sampling.
   *
   * NB the number of runs equals the number of physical allocations the
   * caller will create: a block-aligned fine interleaving (e.g.
   * block_cyclic at exactly the block size) is EXACT but yields one run
   * per block -- the plan is honest about that cost rather than hiding it.
   *
   * SCOPE NOTE: this computes the EXTENSION of the quotient (its runs,
   * enumerated by the certified walk with alignment verified a
   * posteriori), not the quotient as a layout object. A symbolic, leaf-level
   * quotient -- existence decided a priori by congruences on
   * the place-leaf strides/offsets, construction by dividing the strides
   * through, O(leaves) regardless of run count -- is possible future work;
   * what stands in the way is that the allocation is the RESTRICTION of
   * the padded-space layout to the true extents, and that restriction is
   * not a pure layout operation (it is exactly what the row walk performs
   * by hand). Until then the walk's cost is bounded by max_runs and
   * calibrated against the sampled fallback, so the proxy is never a
   * performance regression.
   *
   * @param max_runs walk budget, as in try_block_owners (0 = auto)
   */
  ::cuda::std::optional<::std::vector<block_run>>
  try_block_runs(size_t block_size_bytes, size_t elemsize, size_t max_runs = 0) const
  {
    _CCCL_ASSERT(elemsize > 0 && block_size_bytes >= elemsize, "invalid block geometry");
    const size_t total_bytes = true_dims_.size() * elemsize;
    const size_t nblocks     = (total_bytes + block_size_bytes - 1) / block_size_bytes;
    if (max_runs == 0)
    {
      max_runs = ::cuda::std::max<size_t>(16 * nblocks, size_t(1) << 16);
    }

    ::std::vector<block_run> runs;
    bool aligned = true;
    // the walk aborts on the first misaligned boundary: the remainder tier
    // (try_block_owners) re-walks, so wasting the rest of this walk would
    // double the cost of every remainder allocation for nothing
    const bool completed = for_each_owner_byte_run(elemsize, max_runs, [&](size_t start, size_t len, pos4 o) -> bool {
      if (start % block_size_bytes != 0)
      {
        aligned = false; // internal boundary inside a block: no strict quotient
        return false;
      }
      const size_t first = start / block_size_bytes;
      // the last run's tail may end mid-block; the block is still pure
      const size_t count =
        (::cuda::std::min(start + len, total_bytes) - start + block_size_bytes - 1) / block_size_bytes;
      runs.push_back(block_run{o, first, count});
      return true;
    });
    if (!completed || !aligned)
    {
      return ::cuda::std::nullopt;
    }
    return runs;
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
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition::apply: the task shape does not match the partition's extents");
      }
    }

    ::cuda::std::array<size_t, rank> lo{};
    ::cuda::std::array<size_t, rank> hi{};
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

    ::cuda::std::array<size_t, dims> lo{};
    ::cuda::std::array<size_t, dims> hi{};
    for (size_t d = 0; d < dims; d++)
    {
      const auto lo_bound = b.get_begin(d);
      const auto hi_bound = b.get_end(d);
      if (lo_bound < 0 || lo_bound > hi_bound || static_cast<size_t>(hi_bound) > true_dims_.get(d))
      {
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition::apply: the box is not contained in the partition's extents");
      }
      lo[d] = static_cast<size_t>(lo_bound);
      hi[d] = static_cast<size_t>(hi_bound);
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
      _CCCL_THROW(::std::out_of_range, "cute_partition::place_offset: place index out of range");
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
  int cmp(const cute_partition_descriptor& o) const
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

  bool operator==(const cute_partition_descriptor& o) const
  {
    return cmp(o) == 0;
  }

  bool operator!=(const cute_partition_descriptor& o) const
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
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition::apply: the iteration rank does not match the partition's "
                    "extents");
      }
    }
  }

  template <size_t rank>
  cute_sub_shape<rank> apply_region(
    const ::cuda::std::array<size_t, rank>& lo,
    const ::cuda::std::array<size_t, rank>& hi,
    pos4 place_position,
    dim4 grid_dims) const
  {
    if (!(grid_dims == grid_dims_))
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cute_partition::apply: the grid does not match the partition's grid extents");
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
        _CCCL_THROW(::std::invalid_argument, "cute_partition: place axis out of range");
      }
      if (place_leaves_[k].extent != grid_dims_.get(static_cast<size_t>(a)))
      {
        _CCCL_THROW(::std::invalid_argument, "cute_partition: place leaf extent does not match its grid axis extent");
      }
      for (size_t j = 0; j < k; j++)
      {
        if (place_axes_[j] == a)
        {
          _CCCL_THROW(::std::invalid_argument, "cute_partition: grid axis bound to more than one place leaf");
        }
      }
    }

    // Without replication, every grid axis with extent > 1 must be bound to a
    // tensor dimension; otherwise owner() pins that axis to coordinate 0 and
    // the remaining places on that axis own no bytes. Relax this only if
    // replication is introduced.
    if (num_places() != grid_dims_.size())
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cute_partition: the partition leaves grid places unused (a grid axis with extent > 1 is bound to no "
                  "tensor dimension; replication is not supported). Collapse the unused grid axes or bind them to a "
                  "tensor dimension.");
    }

    for (size_t d = 0; d < 4; d++)
    {
      if (true_dims_.get(d) < 1 || true_dims_.get(d) > padded_dims_.get(d))
      {
        _CCCL_THROW(::std::invalid_argument, "cute_partition: true extents must be within [1, padded extents]");
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
        _CCCL_THROW(::std::invalid_argument, "cute_partition: negative strides are not supported");
      }
      if (l.extent == 0)
      {
        _CCCL_THROW(::std::invalid_argument, "cute_partition: leaf extents must be at least 1");
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
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition: leaves do not tile the padded space exactly (layout must be "
                    "exact and bijective)");
      }
      expected_stride *= all[k].extent;
    }
    if (expected_stride != padded_dims_.size())
    {
      _CCCL_THROW(::std::invalid_argument, "cute_partition: layout size does not match the padded extents");
    }
  }

  struct decode_leaf
  {
    layout_leaf leaf;
    ::cuda::std::ptrdiff_t place_leaf; // index into place_leaves_, or -1 for local leaves
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

//! A tensor dimension that is local to every grid place.
struct whole_dim_spec
{};

//! A tensor dimension split into contiguous blocks over grid axis `axis`.
template <int axis>
struct blocked_dim_spec
{
  static_assert(axis >= 0 && axis < 4, "a partition mesh axis must be in [0, 4)");
};

//! A tensor dimension distributed cyclically over grid axis `axis`.
template <int axis>
struct cyclic_dim_spec
{
  static_assert(axis >= 0 && axis < 4, "a partition mesh axis must be in [0, 4)");
};

//! A tensor dimension distributed in cyclic blocks over grid axis `axis`.
template <int axis>
struct block_cyclic_dim_spec
{
  static_assert(axis >= 0 && axis < 4, "a partition mesh axis must be in [0, 4)");
  size_t block;
};

//! Heterogeneous per-dimension specification preserving partition topology in
//! the C++ type.
template <typename... Specs>
struct partition_spec
{
  explicit partition_spec(Specs... values_)
      : values(mv(values_)...)
  {}

  ::cuda::std::tuple<Specs...> values;
};

template <typename... Specs>
partition_spec(Specs...) -> partition_spec<::cuda::std::decay_t<Specs>...>;

inline constexpr whole_dim_spec whole{};

template <int axis>
inline constexpr blocked_dim_spec<axis> blocked{};

template <int axis>
inline constexpr cyclic_dim_spec<axis> cyclic{};

template <int axis>
block_cyclic_dim_spec<axis> block_cyclic(size_t block)
{
  return {block};
}

template <typename Spec>
struct __partition_spec_traits;

template <>
struct __partition_spec_traits<whole_dim_spec>
{
  static constexpr size_t num_place_leaves = 0;
  static constexpr size_t num_local_leaves = 1;
};

template <int axis>
struct __partition_spec_traits<blocked_dim_spec<axis>>
{
  static constexpr size_t num_place_leaves = 1;
  static constexpr size_t num_local_leaves = 1;
};

template <int axis>
struct __partition_spec_traits<cyclic_dim_spec<axis>>
{
  static constexpr size_t num_place_leaves = 1;
  static constexpr size_t num_local_leaves = 1;
};

template <int axis>
struct __partition_spec_traits<block_cyclic_dim_spec<axis>>
{
  static constexpr size_t num_place_leaves = 1;
  static constexpr size_t num_local_leaves = 2;
};

/**
 * @brief Structured tensor partition with compile-time rank and leaf counts
 *
 * Extents and strides remain runtime values, while exact array extents make
 * the execution descriptor compact and give device code static loop bounds.
 */
template <size_t rank, size_t num_place_leaves, size_t num_local_leaves>
class cute_partition
{
public:
  static_assert(rank > 0 && rank <= 4, "cute_partition rank must be in [1, 4]");

  static constexpr size_t rank_v             = rank;
  static constexpr size_t place_leaf_count_v = num_place_leaves;
  static constexpr size_t local_leaf_count_v = num_local_leaves;

  explicit cute_partition(const cute_partition_descriptor& descriptor)
      : padded_dims_(descriptor.padded_dims())
      , true_dims_(descriptor.true_dims())
      , grid_dims_(descriptor.grid_dims())
  {
    if (descriptor.place_leaves().size() != num_place_leaves || descriptor.place_axes().size() != num_place_leaves
        || descriptor.local_leaves().size() != num_local_leaves)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cute_partition: descriptor topology does not match the static partition type");
    }

    ::cuda::std::copy(descriptor.place_leaves().begin(), descriptor.place_leaves().end(), place_leaves_.begin());
    ::cuda::std::copy(descriptor.place_axes().begin(), descriptor.place_axes().end(), place_axes_.begin());
    ::cuda::std::copy(descriptor.local_leaves().begin(), descriptor.local_leaves().end(), local_leaves_.begin());

    for (size_t d = rank; d < 4; d++)
    {
      if (padded_dims_.get(d) != 1)
      {
        _CCCL_THROW(::std::invalid_argument, "cute_partition: rank does not match the partition extents");
      }
    }
  }

  _CCCL_HOST_DEVICE const dim4& true_dims() const
  {
    return true_dims_;
  }

  _CCCL_HOST_DEVICE const dim4& padded_dims() const
  {
    return padded_dims_;
  }

  _CCCL_HOST_DEVICE const dim4& grid_dims() const
  {
    return grid_dims_;
  }

  _CCCL_HOST_DEVICE ::cuda::std::span<const layout_leaf> place_leaves() const
  {
    return {place_leaves_.data(), num_place_leaves};
  }

  _CCCL_HOST_DEVICE ::cuda::std::span<const int> place_axes() const
  {
    return {place_axes_.data(), num_place_leaves};
  }

  _CCCL_HOST_DEVICE ::cuda::std::span<const layout_leaf> local_leaves() const
  {
    return {local_leaves_.data(), num_local_leaves};
  }

  _CCCL_HOST_DEVICE size_t num_places() const
  {
    size_t result = 1;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t k = 0; k < num_place_leaves; k++)
    {
      result *= place_leaves_[k].extent;
    }
    return result;
  }

  _CCCL_HOST_DEVICE size_t tiles_per_place() const
  {
    size_t result = 1;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t k = 0; k < num_local_leaves; k++)
    {
      result *= local_leaves_[k].extent;
    }
    return result;
  }

  _CCCL_HOST_DEVICE pos4 owner(pos4 data_coords) const
  {
    const size_t linear    = padded_dims_.get_index(data_coords);
    ssize_t place_coord[4] = {0, 0, 0, 0};

    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t k = 0; k < num_place_leaves; k++)
    {
      const layout_leaf& leaf = place_leaves_[k];
      if (leaf.extent > 1)
      {
        const size_t c                                   = (linear / static_cast<size_t>(leaf.stride)) % leaf.extent;
        place_coord[static_cast<size_t>(place_axes_[k])] = static_cast<ssize_t>(c);
      }
    }

    return pos4(place_coord[0], place_coord[1], place_coord[2], place_coord[3]);
  }

  template <typename S>
  auto apply(const S& s, pos4 place_position, dim4 grid_dims) const
  {
    constexpr size_t shape_rank = S::rank();
    static_assert(shape_rank == rank, "the task shape rank must match the cute_partition rank");

    ::cuda::std::array<size_t, rank> lo{};
    ::cuda::std::array<size_t, rank> hi{};
    for (size_t d = 0; d < rank; d++)
    {
      if (static_cast<size_t>(s.extent(d)) != true_dims_.get(d))
      {
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition::apply: the task shape does not match the partition's extents");
      }
      hi[d] = true_dims_.get(d);
    }
    return apply_region(lo, hi, place_position, grid_dims);
  }

  template <size_t dims>
  auto apply(const box<dims>& b, pos4 place_position, dim4 grid_dims) const
  {
    static_assert(dims == rank, "the task box rank must match the cute_partition rank");

    ::cuda::std::array<size_t, rank> lo{};
    ::cuda::std::array<size_t, rank> hi{};
    for (size_t d = 0; d < rank; d++)
    {
      const auto lo_bound = b.get_begin(d);
      const auto hi_bound = b.get_end(d);
      if (lo_bound < 0 || lo_bound > hi_bound || static_cast<size_t>(hi_bound) > true_dims_.get(d))
      {
        _CCCL_THROW(::std::invalid_argument,
                    "cute_partition::apply: the box is not contained in the partition's extents");
      }
      lo[d] = static_cast<size_t>(lo_bound);
      hi[d] = static_cast<size_t>(hi_bound);
    }
    return apply_region(lo, hi, place_position, grid_dims);
  }

  size_t place_offset(size_t place_index) const
  {
    if (place_index >= num_places())
    {
      _CCCL_THROW(::std::out_of_range, "cute_partition::place_offset: place index out of range");
    }

    size_t offset = 0;
    for (size_t k = 0; k < num_place_leaves; k++)
    {
      offset += (place_index % place_leaves_[k].extent) * static_cast<size_t>(place_leaves_[k].stride);
      place_index /= place_leaves_[k].extent;
    }
    return offset;
  }

  cute_partition_descriptor descriptor() const
  {
    return cute_partition_descriptor(
      ::std::vector<layout_leaf>(place_leaves_.begin(), place_leaves_.end()),
      ::std::vector<int>(place_axes_.begin(), place_axes_.end()),
      ::std::vector<layout_leaf>(local_leaves_.begin(), local_leaves_.end()),
      padded_dims_,
      true_dims_,
      grid_dims_);
  }

  int cmp(const cute_partition& other) const
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

    if (const int c = cmp_dims(padded_dims_, other.padded_dims_))
    {
      return c;
    }
    if (const int c = cmp_dims(true_dims_, other.true_dims_))
    {
      return c;
    }
    if (const int c = cmp_dims(grid_dims_, other.grid_dims_))
    {
      return c;
    }
    for (size_t k = 0; k < num_place_leaves; k++)
    {
      if (const int c = cmp_sizes(place_leaves_[k].extent, other.place_leaves_[k].extent))
      {
        return c;
      }
      if (const int c =
            cmp_sizes(static_cast<size_t>(place_leaves_[k].stride), static_cast<size_t>(other.place_leaves_[k].stride)))
      {
        return c;
      }
      if (const int c = cmp_sizes(static_cast<size_t>(place_axes_[k]), static_cast<size_t>(other.place_axes_[k])))
      {
        return c;
      }
    }
    for (size_t k = 0; k < num_local_leaves; k++)
    {
      if (const int c = cmp_sizes(local_leaves_[k].extent, other.local_leaves_[k].extent))
      {
        return c;
      }
      if (const int c =
            cmp_sizes(static_cast<size_t>(local_leaves_[k].stride), static_cast<size_t>(other.local_leaves_[k].stride)))
      {
        return c;
      }
    }
    return 0;
  }

  bool operator==(const cute_partition& other) const
  {
    return cmp(other) == 0;
  }

  bool operator!=(const cute_partition& other) const
  {
    return !(*this == other);
  }

private:
  static_cute_sub_shape<rank, num_local_leaves> apply_region(
    const ::cuda::std::array<size_t, rank>& lo,
    const ::cuda::std::array<size_t, rank>& hi,
    pos4 place_position,
    dim4 grid_dims) const
  {
    if (!(grid_dims == grid_dims_))
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cute_partition::apply: the grid does not match the partition's grid extents");
    }

    const pos4 grid_coords = grid_dims_.index_to_pos(static_cast<size_t>(place_position.x));
    size_t offset          = 0;
    for (size_t k = 0; k < num_place_leaves; k++)
    {
      offset += static_cast<size_t>(grid_coords.get(static_cast<size_t>(place_axes_[k])))
              * static_cast<size_t>(place_leaves_[k].stride);
    }

    return static_cute_sub_shape<rank, num_local_leaves>(local_leaves_, offset, padded_dims_, lo, hi);
  }

  ::cuda::std::array<layout_leaf, num_place_leaves> place_leaves_{};
  ::cuda::std::array<int, num_place_leaves> place_axes_{};
  ::cuda::std::array<layout_leaf, num_local_leaves> local_leaves_{};
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
inline cute_partition_descriptor
make_partition_descriptor(dim4 true_dims, const ::std::vector<dim_spec>& spec, dim4 grid_dims)
{
  if (spec.size() > 4)
  {
    _CCCL_THROW(::std::invalid_argument, "make_partition: at most 4 dimensions are supported");
  }
  const size_t rank = spec.size();

  // Pass 1: padded extent per dimension
  ::cuda::std::array<size_t, 4> padded = {1, 1, 1, 1};
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
      _CCCL_THROW(::std::invalid_argument, "make_partition: mesh_axis out of range");
    }
    const size_t nplaces = grid_dims.get(static_cast<size_t>(e.mesh_axis));
    if (nplaces == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "make_partition: grid axis extents must be at least 1");
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
          _CCCL_THROW(::std::invalid_argument, "make_partition: block_cyclic requires a block size");
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
  ::cuda::std::array<size_t, 4> stride = {1, 1, 1, 1};
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
    if (d >= rank)
    {
      if (padded[d] > 1)
      {
        local_leaves.push_back({padded[d], static_cast<::cuda::std::ptrdiff_t>(R)});
      }
      continue;
    }
    if (spec[d].policy == dim_policy::whole)
    {
      local_leaves.push_back({padded[d], static_cast<::cuda::std::ptrdiff_t>(R)});
      continue;
    }

    const auto& e        = spec[d];
    const size_t nplaces = grid_dims.get(static_cast<size_t>(e.mesh_axis));

    switch (e.policy)
    {
      case dim_policy::blocked: {
        const size_t b = padded[d] / nplaces;
        local_leaves.push_back({b, static_cast<::cuda::std::ptrdiff_t>(R)});
        place_leaves.push_back({nplaces, static_cast<::cuda::std::ptrdiff_t>(b * R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      case dim_policy::cyclic: {
        local_leaves.push_back({padded[d] / nplaces, static_cast<::cuda::std::ptrdiff_t>(nplaces * R)});
        place_leaves.push_back({nplaces, static_cast<::cuda::std::ptrdiff_t>(R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      case dim_policy::block_cyclic: {
        const size_t nsuper = padded[d] / (e.block * nplaces);
        local_leaves.push_back({e.block, static_cast<::cuda::std::ptrdiff_t>(R)});
        local_leaves.push_back({nsuper, static_cast<::cuda::std::ptrdiff_t>(e.block * nplaces * R)});
        place_leaves.push_back({nplaces, static_cast<::cuda::std::ptrdiff_t>(e.block * R)});
        place_axes.push_back(e.mesh_axis);
        break;
      }
      default:
        break;
    }
  }

  return cute_partition_descriptor(
    mv(place_leaves),
    mv(place_axes),
    mv(local_leaves),
    dim4(padded[0], padded[1], padded[2], padded[3]),
    true_dims,
    grid_dims);
}

inline dim_spec __make_runtime_dim_spec(whole_dim_spec)
{
  return {};
}

template <int axis>
dim_spec __make_runtime_dim_spec(blocked_dim_spec<axis>)
{
  return {dim_policy::blocked, axis, 0};
}

template <int axis>
dim_spec __make_runtime_dim_spec(cyclic_dim_spec<axis>)
{
  return {dim_policy::cyclic, axis, 0};
}

template <int axis>
dim_spec __make_runtime_dim_spec(block_cyclic_dim_spec<axis> spec)
{
  return {dim_policy::block_cyclic, axis, spec.block};
}

/**
 * @brief Build a statically shaped partition from typed dimension specs
 */
template <typename... Specs>
auto make_partition(dim4 true_dims, partition_spec<Specs...> spec, dim4 grid_dims)
{
  constexpr size_t rank             = sizeof...(Specs);
  constexpr size_t num_place_leaves = (__partition_spec_traits<Specs>::num_place_leaves + ... + 0);
  constexpr size_t num_local_leaves = (__partition_spec_traits<Specs>::num_local_leaves + ... + 0);
  static_assert(rank > 0 && rank <= 4, "make_partition requires between one and four dimension specs");

  for (size_t d = rank; d < 4; d++)
  {
    if (true_dims.get(d) != 1)
    {
      _CCCL_THROW(::std::invalid_argument, "make_partition: the number of specs does not match the tensor rank");
    }
  }

  ::std::vector<dim_spec> runtime_specs;
  runtime_specs.reserve(rank);
  ::cuda::std::apply(
    [&](const auto&... values) {
      (runtime_specs.push_back(::cuda::experimental::places::__make_runtime_dim_spec(values)), ...);
    },
    spec.values);

  const auto descriptor = ::cuda::experimental::places::make_partition_descriptor(true_dims, runtime_specs, grid_dims);
  return cute_partition<rank, num_place_leaves, num_local_leaves>(descriptor);
}

/**
 * @brief Build an expert statically shaped partition from exact leaf arrays
 */
template <size_t rank, size_t num_place_leaves, size_t num_local_leaves>
cute_partition<rank, num_place_leaves, num_local_leaves> make_partition(
  const ::cuda::std::array<layout_leaf, num_place_leaves>& place_leaves,
  const ::cuda::std::array<int, num_place_leaves>& place_axes,
  const ::cuda::std::array<layout_leaf, num_local_leaves>& local_leaves,
  dim4 padded_dims,
  dim4 true_dims,
  dim4 grid_dims)
{
  const cute_partition_descriptor descriptor(
    ::std::vector<layout_leaf>(place_leaves.begin(), place_leaves.end()),
    ::std::vector<int>(place_axes.begin(), place_axes.end()),
    ::std::vector<layout_leaf>(local_leaves.begin(), local_leaves.end()),
    padded_dims,
    true_dims,
    grid_dims);
  return cute_partition<rank, num_place_leaves, num_local_leaves>(descriptor);
}

/**
 * @brief Evaluate - without allocating - how a localized allocation of a
 * tensor distributed by `partition` over `grid` would be placed
 *
 * See evaluate_localized_placement(); the tensor extents are the partition's
 * true extents.
 */
template <typename Partition>
[[nodiscard]] localized_stats evaluate_localized_placement(
  const exec_place& grid,
  const Partition& partition,
  size_t elemsize,
  size_t probes     = localized_placement_default_probes,
  size_t block_size = 0)
{
  if (!(grid.get_dims() == partition.grid_dims()))
  {
    _CCCL_THROW(::std::invalid_argument, "the partition's grid extents do not match the execution place grid");
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
 * Like data_place_composite but ownership is defined by the partition object
 * value (a bare partition_fn_t cannot carry its leaves), so a canonical
 * descriptor is stored on the place. Because a padded partition is
 * intrinsically specific to one tensor, such a place is per-tensor by nature.
 * This place supports both shaped raw allocations
 * (allocate_nd(data_dims, elemsize)) and STF logical data.
 */

/**
 * @brief Owner provider for localized_array from a partition descriptor.
 *
 * Tries the analytic block plan first (exact owners, byte-true placement
 * statistics: the sample counters then hold byte counts); falls back to the
 * sampled majority vote for layouts denser than the placement blocks.
 */
inline auto make_partition_placement_provider(
  const cute_partition_descriptor& partition, dim4 data_dims, size_t total_size, size_t elemsize)
{
  return [partition, data_dims, total_size, elemsize](
           size_t block_size_bytes, size_t nblocks, localized_stats& stats) -> ::std::vector<block_run> {
    // Budget the analytic walks against what the sampled fallback would
    // spend anyway (probes owner() evaluations per block): when a walk fits
    // this budget it is BOTH cheaper and exact, so choosing it can never be
    // a performance regression. The floor keeps small allocations
    // permissive.
    const size_t budget = ::cuda::std::max<size_t>(nblocks * localized_placement_default_probes, size_t(1) << 16);

    // Exact tier: the strict quotient exists -- runs come straight from the
    // layout algebra, no per-block work at all.
    if (auto runs = partition.try_block_runs(block_size_bytes, elemsize, budget))
    {
      stats.total_samples    = total_size * elemsize;
      stats.matching_samples = stats.total_samples; // exact: zero misplacement
      return mv(*runs);
    }
    // Census tier: straddling blocks resolved by exact byte majority with a
    // closed-form misplaced count (byte-true accuracy in the stats).
    size_t misplaced = 0;
    if (auto owners = partition.try_block_owners(block_size_bytes, elemsize, &misplaced, budget))
    {
      stats.total_samples    = total_size * elemsize;
      stats.matching_samples = stats.total_samples - misplaced;
      return owners_to_block_runs(mv(*owners));
    }
    // Sampled tier: opaque-density fallback (element-pitch interleavings).
    const auto owner_of = ::std::function<pos4(size_t)>([&partition, data_dims](size_t ind) {
      return partition.owner(data_dims.index_to_pos(ind));
    });
    return owners_to_block_runs(compute_block_owners(
      owner_of, nblocks, block_size_bytes, elemsize, total_size, localized_placement_default_probes, stats));
  };
}

class data_place_cute_composite final : public data_place_interface
{
public:
  data_place_cute_composite(exec_place grid, cute_partition_descriptor partition)
      : grid_(mv(grid))
      , partition_(mv(partition))
  {
    if (!(grid_.get_dims() == partition_.grid_dims()))
    {
      _CCCL_THROW(::std::invalid_argument, "the partition's grid extents do not match the execution place grid");
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
    _CCCL_THROW(::std::logic_error, "hash() not supported for composite data_place");
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

  void* allocate(::cuda::std::ptrdiff_t, cudaStream_t) const override
  {
    _CCCL_THROW(::std::runtime_error,
                "cute-partition composite data_place cannot allocate from a byte count alone: use allocate_nd with the "
                "partition's true extents or allocate through logical data");
  }

  void* allocate_nd(dim4 data_dims, size_t elemsize, cudaStream_t) const override
  {
    // A padded partition is specific to one tensor: the requested extents
    // must be the ones the partition was built for.
    if (!(data_dims == partition_.true_dims()))
    {
      _CCCL_THROW(::std::invalid_argument,
                  "cute composite data_place: requested extents do not match the partition's true "
                  "extents");
    }

    auto arr = ::std::make_unique<localized_array>(
      grid_,
      make_partition_placement_provider(partition_, data_dims, data_dims.size(), elemsize),
      data_dims.size(),
      elemsize,
      data_dims);
    void* ptr                           = arr->get_base_ptr();
    get_composite_alloc_registry()[ptr] = ::cuda::std::move(arr);
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

  const cute_partition_descriptor& get_partition() const
  {
    return partition_;
  }

  const exec_place& get_grid() const
  {
    return grid_;
  }

private:
  exec_place grid_;
  cute_partition_descriptor partition_;
};

/**
 * @brief Create a composite data place backed by an erased partition
 */
inline data_place make_composite_data_place(const exec_place& grid, cute_partition_descriptor partition)
{
  return data_place(::std::make_shared<data_place_cute_composite>(grid, mv(partition)));
}

/**
 * @brief Create a composite data place from a typed partition
 */
template <size_t rank, size_t num_place_leaves, size_t num_local_leaves>
data_place make_composite_data_place(const exec_place& grid,
                                     const cute_partition<rank, num_place_leaves, num_local_leaves>& partition)
{
  return ::cuda::experimental::places::make_composite_data_place(grid, partition.descriptor());
}

#ifdef UNITTESTED_FILE
UNITTEST("make_partition blocked leaves and owners")
{
  // 2-D tensor (6, 4), dimension 1 blocked over 2 places (axis 0)
  const dim4 true_dims(6, 4);
  const dim4 grid_dims(2);
  auto part = make_partition(true_dims, partition_spec{whole, blocked<0>}, grid_dims);

  static_assert(decltype(part)::rank_v == 2);
  static_assert(decltype(part)::place_leaf_count_v == 1);
  static_assert(decltype(part)::local_leaf_count_v == 2);
  static_assert(::cuda::std::is_trivially_copyable_v<decltype(part)>);
  static_assert(sizeof(static_cute_sub_shape<2, 2>) < sizeof(cute_sub_shape<2>));

  const auto runtime_part =
    make_partition_descriptor(true_dims, {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, grid_dims);
  EXPECT(part.descriptor() == runtime_part);

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
  auto part = make_partition(true_dims, partition_spec{whole, blocked<0>}, grid_dims);

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

  auto cyc = make_partition(dim4(7), partition_spec{cyclic<0>}, grid_dims);
  for (size_t x = 0; x < 7; x++)
  {
    EXPECT(cyc.owner(pos4(x)) == pos4(x % 2));
  }

  auto bc = make_partition(dim4(8), partition_spec{block_cyclic<0>(2)}, grid_dims);
  static_assert(decltype(bc)::local_leaf_count_v == 2);
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
  auto part = make_partition(true_dims, partition_spec{blocked<0>}, grid_dims);

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
  const cute_partition_descriptor a({{2, 1}}, {0}, {{6, 2}}, dim4(2, 6), true_dims, grid_dims);
  const cute_partition_descriptor b({{2, 1}}, {0}, {{6, 2}}, dim4(3, 4), true_dims, grid_dims);

  EXPECT(a != b);
  EXPECT(!(a.owner(pos4(0, 1)) == b.owner(pos4(0, 1))));
};

UNITTEST("cute_partition rejects lower-rank iteration")
{
  // Dimension 1 has a true extent of one but is padded to two by its split;
  // omitting it would still discard ownership coordinates and duplicate work.
  const dim4 true_dims(8, 1);
  const dim4 grid_dims(2, 2);
  const auto part = make_partition_descriptor(
    true_dims, {dim_spec{dim_policy::blocked, 0, 0}, dim_spec{dim_policy::blocked, 1, 0}}, grid_dims);
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

UNITTEST("cute partitions reject reversed regions")
{
  const dim4 dims(8);
  const dim4 grid(2);
  const auto part       = make_partition(dims, partition_spec{blocked<0>}, grid);
  const auto descriptor = part.descriptor();
  const box<1> reversed({6ul, 2ul});

  const auto rejects = [&](const auto& partition) {
    try
    {
      (void) partition.apply(reversed, pos4(0), grid);
    }
    catch (const ::std::invalid_argument&)
    {
      return true;
    }
    return false;
  };

  EXPECT(rejects(part));
  EXPECT(rejects(descriptor));
};

UNITTEST("cute_partition validation rejects inexact layouts")
{
  const dim4 dims(8);
  const dim4 grid(2);

  const ::cuda::std::array<layout_leaf, 1> place_leaves{{{2, 4}}};
  const ::cuda::std::array<int, 1> place_axes{{0}};
  const ::cuda::std::array<layout_leaf, 1> local_leaves{{{4, 1}}};
  const auto exact = make_partition<1>(place_leaves, place_axes, local_leaves, dims, dims, grid);
  EXPECT(exact.owner(pos4(7)) == pos4(1));

  // Overlapping: both leaves have stride 1
  bool thrown = false;
  try
  {
    cute_partition_descriptor({{2, 1}}, {0}, {{4, 1}}, dims, dims, grid);
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
    cute_partition_descriptor({{2, 4}}, {0}, {{2, 1}}, dims, dims, grid);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);
};

UNITTEST("min_owner_run_bytes reports the certified run granularity")
{
  // blocked: the smallest place-leaf stride is the chunk -> chunk * elemsize
  const auto blocked_part = make_partition_descriptor(dim4(16), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  EXPECT(blocked_part.min_owner_run_bytes(4) == 8 * 4);

  // cyclic: ownership can change every element -> elemsize
  const auto cyclic_part = make_partition_descriptor(dim4(16), {dim_spec{dim_policy::cyclic, 0, 0}}, dim4(2));
  EXPECT(cyclic_part.min_owner_run_bytes(4) == 4);

  // block_cyclic(b): ownership changes at block pitch -> b * elemsize
  const auto bc_part = make_partition_descriptor(dim4(32), {dim_spec{dim_policy::block_cyclic, 0, 4}}, dim4(2));
  EXPECT(bc_part.min_owner_run_bytes(2) == 4 * 2);

  // no place mode (nothing distributed): 0 by convention
  const auto whole_part = make_partition_descriptor(dim4(16), {dim_spec{}}, dim4(1));
  EXPECT(whole_part.min_owner_run_bytes(4) == 0);

  // consistency: a run granularity at or above the block size admits an
  // analytic plan; far below it (with a tight budget) it declines
  size_t mis           = 0;
  const auto fine_plan = cyclic_part.try_block_owners(64, 4, &mis); // small case: still analyzable
  EXPECT(fine_plan.has_value() == true);
};

UNITTEST("extent-1 place leaves carry no ownership boundary")
{
  // expert-form descriptor with a degenerate {extent 1, stride 0} place
  // leaf: owner() ignores it, so the granularity and the walks must too
  // (a raw minimum over strides would divide by zero downstream).
  const dim4 dims(16);
  const auto part = cute_partition_descriptor({{1, 0}}, /* axes */ {0}, /* local */ {{16, 1}}, dims, dims, dim4(1));
  EXPECT(part.min_owner_run_bytes(4) == 0);
  size_t misplaced  = ~size_t(0);
  const auto owners = part.try_block_owners(8, 4, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(misplaced == 0);
  for (const auto& o : *owners)
  {
    EXPECT(o == pos4(0));
  }
  const auto runs = part.try_block_runs(8, 4);
  EXPECT(runs.has_value() == true);
  EXPECT(runs->size() == 1);
};

UNITTEST("try_block_runs: strict quotient emitted directly as runs")
{
  // 16 elements of 4 B blocked over 2 places, 8 B blocks: boundary at byte
  // 32 = block 4 -> two runs, no per-block materialization.
  const auto part = make_partition_descriptor(dim4(16), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  const auto runs = part.try_block_runs(8, 4);
  EXPECT(runs.has_value() == true);
  EXPECT(runs->size() == 2);
  EXPECT((*runs)[0].owner == pos4(0));
  EXPECT((*runs)[0].first_block == 0);
  EXPECT((*runs)[0].num_blocks == 4);
  EXPECT((*runs)[1].owner == pos4(1));
  EXPECT((*runs)[1].first_block == 4);
  EXPECT((*runs)[1].num_blocks == 4);
};

UNITTEST("try_block_runs: declines when a boundary falls inside a block")
{
  // 13 one-byte elements blocked over 2: boundary at byte 7, blocks of 4 ->
  // no strict quotient; the census tier (try_block_owners) handles it.
  const auto part = make_partition_descriptor(dim4(13), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  EXPECT(!part.try_block_runs(4, 1).has_value());
};

UNITTEST("try_block_runs: single place covers the VM tail")
{
  // 10 elements of 4 B on one place, 16 B blocks: 40 B of payload round up
  // to 3 blocks; the single run must cover the partial tail block too.
  const auto part = make_partition_descriptor(dim4(10), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(1));
  const auto runs = part.try_block_runs(16, 4);
  EXPECT(runs.has_value() == true);
  EXPECT(runs->size() == 1);
  EXPECT((*runs)[0].first_block == 0);
  EXPECT((*runs)[0].num_blocks == 3);
};

UNITTEST("try_block_owners: exact plan when boundaries align")
{
  // 16 elements of 4 B blocked over 2 places, 8 B blocks: the ownership
  // boundary (element 8 == byte 32) is block-aligned -> exact plan.
  const auto part   = make_partition_descriptor(dim4(16), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  size_t misplaced  = ~size_t(0);
  const auto owners = part.try_block_owners(/* block */ 8, /* elemsize */ 4, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(misplaced == 0); // factors through blocks: no placement error
  EXPECT(owners->size() == 8);
  for (size_t b = 0; b < 8; b++)
  {
    EXPECT((*owners)[b] == pos4(b < 4 ? 0 : 1));
  }
};

UNITTEST("try_block_owners: straddling block goes to the byte majority")
{
  // 13 one-byte elements blocked over 2 places: chunk = ceil(13/2) = 7, so
  // the boundary falls at byte 7. With 4 B blocks, block 1 (bytes 4..7)
  // holds 3 bytes of place 0 and 1 byte of place 1 -> majority place 0,
  // exactly 1 misplaced byte; every other block is pure.
  const auto part   = make_partition_descriptor(dim4(13), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  size_t misplaced  = 0;
  const auto owners = part.try_block_owners(4, 1, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(misplaced == 1);
  EXPECT(owners->size() == 4);
  EXPECT((*owners)[0] == pos4(0));
  EXPECT((*owners)[1] == pos4(0)); // straddle: 3 bytes p0 vs 1 byte p1
  EXPECT((*owners)[2] == pos4(1));
  EXPECT((*owners)[3] == pos4(1)); // tail block (bytes 12..12), pure p1
};

UNITTEST("try_block_owners: dense element-cyclic declines (sampled fallback)")
{
  // Element-cyclic over 2 places interleaves owners at element pitch: far
  // below any realistic block size, and the run enumeration would need one
  // run per element -> the analytic plan must decline via max_runs.
  const auto part  = make_partition_descriptor(dim4(1 << 20), {dim_spec{dim_policy::cyclic, 0, 0}}, dim4(2));
  size_t misplaced = 0;
  EXPECT(!part.try_block_owners(1 << 16, 4, &misplaced, /* max_runs */ 1 << 10).has_value());
};

UNITTEST("try_block_owners: coarse block_cyclic is analyzable and majority-correct")
{
  // block_cyclic(4) over 2 places, 4 B elements, 32 B blocks: each block
  // holds two 16 B owner runs (4 elements each) -> every block is a 50/50
  // straddle; majority tie-breaks deterministically and the misplaced count
  // is exactly half the payload.
  const auto part   = make_partition_descriptor(dim4(32), {dim_spec{dim_policy::block_cyclic, 0, 4}}, dim4(2));
  size_t misplaced  = 0;
  const auto owners = part.try_block_owners(32, 4, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(owners->size() == 4);
  EXPECT(misplaced == 32 * 4 / 2);
};

UNITTEST("try_block_owners: single place is one exact run")
{
  const auto part = make_partition_descriptor(dim4(1000), {dim_spec{dim_policy::blocked, 0, 0}}, dim4(1));
  // a unit grid axis contributes an extent-1 place leaf, which carries no
  // ownership boundary: the certified granularity must report "no
  // boundary" (0), not collapse to the leaf's stride
  EXPECT(part.min_owner_run_bytes(4) == 0);
  size_t misplaced  = 0;
  const auto owners = part.try_block_owners(64, 4, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(misplaced == 0);
  for (const auto& o : *owners)
  {
    EXPECT(o == pos4(0));
  }
};

UNITTEST("try_block_owners: 2-D expert-major rows stay exact")
{
  // (8 x 64) 4 B tensor, dimension 0 blocked over 2 places, rows of 256 B:
  // with 256 B blocks each row IS a block -> exact, first half place 0.
  // NB dimension 0 varies fastest in the linearization, so "expert-major"
  // here means dim 1 indexes the expert row.
  const auto part  = make_partition_descriptor(dim4(64, 8), {dim_spec{}, dim_spec{dim_policy::blocked, 0, 0}}, dim4(2));
  size_t misplaced = 0;
  const auto owners = part.try_block_owners(256, 4, &misplaced);
  EXPECT(owners.has_value() == true);
  EXPECT(misplaced == 0);
  EXPECT(owners->size() == 8);
  for (size_t b = 0; b < 8; b++)
  {
    EXPECT((*owners)[b] == pos4(b < 4 ? 0 : 1));
  }
};

UNITTEST("make_partition rejects partitions that leave grid places unused")
{
  const dim4 true_dims(64);
  const dim4 grid_dims(6, 4);

  // Blocked on axis 0 only: 6 of 24 places would own data; axes 1..3 idle.
  bool thrown = false;
  try
  {
    make_partition(true_dims, partition_spec{blocked<0>}, grid_dims);
  }
  catch (const ::std::invalid_argument&)
  {
    thrown = true;
  }
  EXPECT(thrown);

  // Binding every grid axis uses all places.
  auto part = make_partition(dim4(12, 8), partition_spec{blocked<0>, blocked<1>}, grid_dims);
  EXPECT(part.num_places() == grid_dims.size());
};
#endif // UNITTESTED_FILE
} // namespace cuda::experimental::places
