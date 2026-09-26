//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief PROTOTYPE — spec→view bind over a contiguous buffer, and the STF
 *        task adapters that hand a sharded view of the i-th task argument to
 *        a task body.
 *
 * Nothing here touches the sharded concepts. Three pieces:
 *
 *  - `make_contiguous_cut` / `bind_view`: apply a `partition_mapper` to a
 *    1-D extent and produce one shard per grid place over a base pointer —
 *    the LOGICAL cut (element granularity, element-size independent), never
 *    the physical VMM runs. Refuses partitions that are not contiguous per
 *    place (cyclic).
 *  - `task_view<T>(task, i, instance)`: the sharded view of the i-th task
 *    argument. Contract: the instance's pieces map one-to-one onto the
 *    task's places, or it throws (replicated; composite over another grid;
 *    single-place instance under a grid task; rank > 1).
 *  - `task_envs(task, view)`: a LAZY per-place environment range built from
 *    the task's own streams — `size()` is the shard count, `operator[]`
 *    constructs the env on access. Nothing is materialized.
 *
 * Duck-typed on the task (`get_task_deps()`, `get_exec_place()`,
 * `get_stream(i)`, `get_stream()`) so this header does not include STF.
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

#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/container/shard.cuh>
#include <cuda/experimental/__sharded/container/sharded_array.cuh>

#include <stdexcept>
#include <string>
#include <vector>

namespace cuda::experimental::sharded
{

/// @brief Per-place contiguous element ranges of a 1-D partition.
struct contiguous_cut
{
  ::std::vector<size_t> begin;
  ::std::vector<size_t> end;
};

namespace reserved
{
inline size_t
__owner_index(const places::partition_mapper& mapper, size_t i, const places::dim4& data_dims, const places::dim4& grid_dims)
{
  places::pos4 p(0);
  mapper(&p, data_dims.index_to_pos(i), data_dims, grid_dims);
  return grid_dims.get_index(p);
}
} // namespace reserved

/**
 * @brief The logical cut of @p mapper over @p n elements on @p grid: for each
 * grid place, the contiguous [begin, end) it owns.
 *
 * Ownership is assumed monotone in the element index (blocked, tiled with one
 * tile per place, whole); each boundary is found by binary search, then
 * verified by sampling. A partition whose ownership is not contiguous per
 * place (cyclic, block-cyclic) is refused: it has no one-shard-per-place
 * representation.
 *
 * @throws std::invalid_argument when the partition is not contiguous per place.
 */
inline contiguous_cut make_contiguous_cut(
  const places::partition_mapper& mapper, size_t n, const places::exec_place& grid, const char* what = "sharded::bind_view")
{
  const places::dim4 data_dims(n);
  const places::dim4 grid_dims = grid.get_dims();
  const size_t P               = grid.size();

  contiguous_cut cut;
  cut.begin.assign(P, 0);
  cut.end.assign(P, 0);
  if (n == 0 || P == 0)
  {
    return cut;
  }

  auto owner = [&](size_t i) {
    return reserved::__owner_index(mapper, i, data_dims, grid_dims);
  };

  // begin[p] = first element whose owner index is >= p (monotone assumption)
  for (size_t p = 0; p < P; ++p)
  {
    size_t lo = 0, hi = n;
    while (lo < hi)
    {
      const size_t mid = lo + (hi - lo) / 2;
      if (owner(mid) < p)
      {
        lo = mid + 1;
      }
      else
      {
        hi = mid;
      }
    }
    cut.begin[p] = lo;
  }
  for (size_t p = 0; p < P; ++p)
  {
    cut.end[p] = (p + 1 < P) ? cut.begin[p + 1] : n;
  }

  // Verify the monotone assumption by sampling each range.
  for (size_t p = 0; p < P; ++p)
  {
    const size_t b = cut.begin[p], e = cut.end[p];
    if (e < b)
    {
      _CCCL_THROW(::std::invalid_argument, ::std::string(what) + ": partition ownership is not monotone in the element index");
    }
    constexpr size_t samples = 16;
    for (size_t k = 0; k < samples && b < e; ++k)
    {
      const size_t i = b + ((e - b - 1) * k) / (samples - 1);
      if (owner(i) != p)
      {
        _CCCL_THROW(::std::invalid_argument,
                    ::std::string(what) + ": partition is not contiguous per place (element " + ::std::to_string(i)
                      + " is owned by place " + ::std::to_string(owner(i)) + ", expected " + ::std::to_string(p)
                      + "); cyclic layouts have no one-shard-per-place view");
      }
    }
  }
  return cut;
}

/**
 * @brief Bind a partition to a contiguous buffer: one shard per grid place at
 * the logical cut, executed by that place on @p streams[p]. Zero-copy; the
 * caller owes the buffer's lifetime. Placement is DECLARED (each piece is
 * assumed to live at its place's affine data place); see
 * `composite_backing` to verify against a VMM-placed allocation.
 */
template <class _Tp>
[[nodiscard]] sharded_array<_Tp> bind_view(
  _Tp* base,
  size_t n,
  const places::partition_mapper& mapper,
  const places::exec_place& grid,
  const ::std::vector<cudaStream_t>& streams,
  const char* what = "sharded::bind_view")
{
  const size_t P = grid.size();
  if (streams.size() != P)
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(what) + ": " + ::std::to_string(streams.size()) + " streams for a grid of "
                  + ::std::to_string(P) + " places");
  }
  const contiguous_cut cut = make_contiguous_cut(mapper, n, grid, what);

  ::std::vector<shard<_Tp>> shards(P);
  for (size_t p = 0; p < P; ++p)
  {
    shard<_Tp>& s   = shards[p];
    s.data          = base + cut.begin[p];
    s.size          = cut.end[p] - cut.begin[p];
    s.capacity      = s.size;
    s.global_offset = cut.begin[p];
    s.exec          = grid.get_place(p);
    s.place         = s.exec.affine_data_place();
    s.stream        = streams[p];
  }
  return sharded_array<_Tp>::adopt(::std::move(shards));
}

/// @brief The VMM-placed backing of a composite-place allocation, if @p base
/// is one (registry lookup), else nullptr. Lets a caller VERIFY declared
/// placement: `backing->get_stats()` reports bytes per place and accuracy.
[[nodiscard]] inline const places::localized_array* composite_backing(const void* base)
{
  auto& reg     = places::get_composite_alloc_registry();
  const auto it = reg.find(const_cast<void*>(base));
  return it == reg.end() ? nullptr : it->second.get();
}

// ===========================================================================
// STF task adapters (duck-typed on the task)
// ===========================================================================

/**
 * @brief The sharded view of the @p dep_index-th argument of @p t, whose
 * instance in the body is @p inst (an mdspan-like slice: `element_type`,
 * `data_handle()`, `size()`, `rank()`). The element type is deduced from the
 * instance.
 *
 * | instance's data place                 | result                              |
 * |---------------------------------------|-------------------------------------|
 * | single place, task on one place       | one shard on that place             |
 * | composite over the task's grid        | the logical cut                     |
 * | composite over another grid           | throws                              |
 * | replicated                            | throws                              |
 * | single place, task on a grid          | throws                              |
 * | rank > 1                              | static_assert (column blocks: phase-2) |
 *
 * PROTOTYPE: const-ness of read-only instances is dropped (`sharded_array`
 * has no const-element form yet).
 */
template <class _Task, class _Slice>
[[nodiscard]] auto task_view(_Task& t, size_t dep_index, const _Slice& inst, const char* what = "sharded::task_view")
{
  // The element type comes from the instance (a read-only instance is
  // `slice<const T>`; the view drops the const, see above).
  using _Tp = ::cuda::std::remove_const_t<typename _Slice::element_type>;
  using ::std::to_string;
  static_assert(_Slice::rank() == 1,
                "sharded::task_view: only rank-1 instances have a contiguous per-place view (column blocks need "
                "the phase-2 packed materialization)");

  const auto& deps = t.get_task_deps();
  if (dep_index >= deps.size())
  {
    _CCCL_THROW(::std::out_of_range,
                ::std::string(what) + ": argument " + to_string(dep_index) + " of a task with " + to_string(deps.size())
                  + " dependencies");
  }
  const places::data_place& dp     = deps[dep_index].get_dplace();
  const places::exec_place& eplace = t.get_exec_place();
  _Tp* base                        = const_cast<_Tp*>(inst.data_handle());
  const size_t n                   = static_cast<size_t>(inst.size());

  if (!dp.is_invalid() && dp.is_replicated())
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(what) + ": argument " + to_string(dep_index)
                  + " is replicated; a replicated instance has no sharded view (whole-per-place is a different shape)");
  }

  if (dp.is_invalid() || !dp.is_composite())
  {
    if (eplace.size() != 1)
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": argument " + to_string(dep_index)
                    + " lives on a single place but the task runs on a grid of " + to_string(eplace.size())
                    + " places; give the dependency a composite data place over the task's grid");
    }
    shard<_Tp> s;
    s.data          = base;
    s.size          = n;
    s.capacity      = n;
    s.global_offset = 0;
    s.exec          = eplace;
    s.place         = dp.is_invalid() ? eplace.affine_data_place() : dp;
    s.stream        = t.get_stream();
    return sharded_array<_Tp>::adopt(::std::vector<shard<_Tp>>{s});
  }

  const auto* comp               = static_cast<const places::data_place_composite*>(dp.get_impl().get());
  const places::exec_place& grid = comp->get_grid();
  if (!(grid == eplace))
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(what) + ": argument " + to_string(dep_index) + " is composite over grid "
                  + grid.to_string() + " but the task runs on " + eplace.to_string()
                  + "; a view over another grid must be built explicitly (sharded::bind_view)");
  }

  const size_t P = grid.size();
  ::std::vector<cudaStream_t> streams(P);
  for (size_t p = 0; p < P; ++p)
  {
    streams[p] = t.get_stream(p);
  }
  return bind_view<_Tp>(base, n, comp->get_partitioner(), grid, streams, what);
}

/**
 * @brief A lazy per-place environment range over a task: env g is built on
 * access from the task's stream for the shard's grid position and that
 * place's affine memory resource. Models `sharded_env_range` (and the
 * allocating variant) without materializing anything; K shards on P places
 * resolve to P distinct envs.
 */
template <class _Task>
struct task_env_range
{
  _Task* task = nullptr;
  ::std::vector<size_t> place_of_shard; // shard g -> grid position

  [[nodiscard]] size_t size() const noexcept
  {
    return place_of_shard.size();
  }

  [[nodiscard]] auto operator[](size_t g) const
  {
    const size_t p = place_of_shard[g];
    return places::place_group::env(task->get_exec_place().get_place(p).affine_data_place(), task->get_stream(p));
  }
};

/// @brief Build the lazy env range of @p view over @p t: each shard's exec
/// place is matched to a grid position of the task by identity.
/// @throws std::invalid_argument when a shard's place is not one of the task's.
template <class _Task, class _View>
[[nodiscard]] task_env_range<_Task> task_envs(_Task& t, const _View& view, const char* what = "sharded::task_envs")
{
  const places::exec_place& grid = t.get_exec_place();
  const size_t P                 = grid.size();
  const size_t K                 = view.num_shards();
  task_env_range<_Task> r;
  r.task = &t;
  r.place_of_shard.resize(K);
  for (size_t g = 0; g < K; ++g)
  {
    bool found = false;
    for (size_t p = 0; p < P && !found; ++p)
    {
      if (grid.get_place(p) == view.shard(g).exec)
      {
        r.place_of_shard[g] = p;
        found               = true;
      }
    }
    if (!found)
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": shard " + ::std::to_string(g) + " executes on "
                    + view.shard(g).exec.to_string() + ", which is not a place of the task's grid " + grid.to_string());
    }
  }
  return r;
}

} // namespace cuda::experimental::sharded
