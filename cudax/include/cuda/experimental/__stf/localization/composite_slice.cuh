//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Async caching layer for localized_array allocations, used by
 *        the STF backends to recycle composite VMM allocations.
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

#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__places/cute_partition.cuh>
#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/internal/stf_places_extended_exports.cuh>
#include <cuda/experimental/__stf/internal/stf_places_into_stf_core.cuh>

#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuda::experimental::stf::reserved
{
/*!
 * @brief A simple object pool with linear search for managing objects of type `T`.
 *
 * The `linear_pool` class provides a basic mechanism for reusing objects of a
 * specific type. It stores a collection of objects and allows retrieval of
 * existing objects with matching parameters or creation of new objects if
 * necessary.
 *
 * @tparam T The type of objects to be managed by the pool.
 */
template <class T>
class linear_pool
{
public:
  linear_pool() = default;

  void put(::std::unique_ptr<T> p)
  {
    EXPECT(p);
    payload.push_back(mv(p));
  }

  template <typename... P>
  ::std::unique_ptr<T> get(P&&... p)
  {
    for (auto it = payload.begin(); it != payload.end(); ++it)
    {
      T* e = it->get();
      assert(e);
      if (*e == ::cuda::std::tuple<const P&...>(p...))
      {
        it->release();
        if (it + 1 < payload.end())
        {
          *it = mv(payload.back());
        }
        payload.pop_back();
        return ::std::unique_ptr<T>(e);
      }
    }

    return ::std::make_unique<T>(::cuda::std::forward<P>(p)...);
  }

  template <typename F>
  void each(F&& f)
  {
    for (auto& ptr : payload)
    {
      assert(ptr);
      f(*ptr);
    }
  }

  //! Move every entry of \p other into this pool (\p other is left in a
  //! moved-from state)
  void import_from(linear_pool&& other)
  {
    payload.insert(
      payload.end(), ::std::make_move_iterator(other.payload.begin()), ::std::make_move_iterator(other.payload.end()));
  }

private:
  ::std::vector<::std::unique_ptr<T>> payload;
};

/**
 * @brief Pairs a localized_array with an event_list for async cache reuse.
 *
 * When a localized_array is returned to the cache after deallocation, we
 * record the outstanding prereqs so that the next consumer waits for them
 * before reusing the VMM allocation.
 */
struct cached_localized_array
{
  explicit cached_localized_array(::std::unique_ptr<localized_array> arr)
      : array(mv(arr))
  {}

  template <typename... Args,
            typename = ::cuda::std::enable_if_t<::cuda::std::is_constructible_v<localized_array, Args...>>>
  explicit cached_localized_array(Args&&... args)
      : array(::std::make_unique<localized_array>(::cuda::std::forward<Args>(args)...))
  {}

  template <typename... P>
  bool operator==(::cuda::std::tuple<P&...> t) const
  {
    return *array == t;
  }

  ::std::unique_ptr<localized_array> array;
  event_list prereqs;
};

/**
 * @brief Cached localized array whose placement is described by an erased
 *        cute_partition value.
 *
 * A cute_partition's ownership mapping is defined by the object value, not
 * just its static topology. Keep the canonical descriptor with the cached
 * allocation so independently constructed equivalent composite places can
 * reuse the same VMM mapping.
 */
struct cached_cute_localized_array
{
  template <typename F>
  explicit cached_cute_localized_array(
    exec_place grid_,
    ::cuda::experimental::places::cute_partition_descriptor partition_,
    F&& delinearize,
    size_t total_size,
    size_t elem_size,
    dim4 data_dims_)
      : grid(mv(grid_))
      , partition(mv(partition_))
      , total_size_bytes(total_size * elem_size)
      , data_dims(data_dims_)
      , elemsize(elem_size)
  {
    const auto owner_of = ::std::function<pos4(size_t)>(
      [partition = this->partition, delinearize = ::cuda::std::forward<F>(delinearize)](size_t ind) {
        return partition.owner(delinearize(ind));
      });
    array = ::std::make_unique<localized_array>(grid, owner_of, total_size, elem_size, data_dims);
  }

  explicit cached_cute_localized_array(
    exec_place grid_,
    ::cuda::experimental::places::cute_partition_descriptor partition_,
    size_t total_size,
    size_t elem_size,
    dim4 data_dims_,
    ::std::unique_ptr<localized_array> array_)
      : grid(mv(grid_))
      , partition(mv(partition_))
      , total_size_bytes(total_size * elem_size)
      , data_dims(data_dims_)
      , elemsize(elem_size)
      , array(mv(array_))
  {}

  template <typename... P>
  bool operator==(::cuda::std::tuple<P&...> t) const
  {
    // tuple arguments:
    // 0: grid, 1: partition, 2: delinearize function, 3: total size,
    // 4: element size, 5: data dimensions
    return grid == ::cuda::std::get<0>(t) && partition == ::cuda::std::get<1>(t)
        && total_size_bytes == ::cuda::std::get<3>(t) * ::cuda::std::get<4>(t) && elemsize == ::cuda::std::get<4>(t)
        && data_dims == ::cuda::std::get<5>(t);
  }

  exec_place grid;
  ::cuda::experimental::places::cute_partition_descriptor partition;
  size_t total_size_bytes;
  dim4 data_dims;
  size_t elemsize;
  ::std::unique_ptr<localized_array> array;
  event_list prereqs;
};

//! Padded per-replica stride of a replicated instance: the replica size
//! rounded up to the placement granularity, so each replica's pages can be
//! bound to its own place. The single source of truth shared by allocation
//! (composite cache) and the per-place instance rebase in parallel_for.
inline size_t replicated_replica_stride(size_t bytes)
{
  const size_t g = ::cuda::experimental::places::default_placement_block_size();
  return ((bytes + g - 1) / g) * g;
}

/**
 * @brief Pairs the replicated localized_array of one (grid, stride) with an
 * event_list for async cache reuse (same lifecycle as the composite
 * entries: the VMM teardown is synchronous, so arrays are recycled through
 * the cache and handed to the parent context on stackable pops).
 */
struct cached_replicated_array
{
  //! Build the replicas: one padded copy per grid member, place-local
  cached_replicated_array(const exec_place& grid_, const size_t& stride_bytes_)
      : grid(grid_)
      , stride_bytes(stride_bytes_)
  {
    const size_t nplaces = grid.size();
    const size_t stride  = stride_bytes;
    array                = ::std::make_unique<localized_array>(
      grid,
      [stride](size_t byte_index) {
        return pos4(static_cast<ssize_t>(byte_index / stride));
      },
      nplaces * stride,
      1,
      dim4(nplaces * stride));
  }

  //! Wrap an existing array returned to the cache
  cached_replicated_array(const exec_place& grid_, size_t stride_bytes_, ::std::unique_ptr<localized_array> a)
      : grid(grid_)
      , stride_bytes(stride_bytes_)
      , array(mv(a))
  {}

  bool operator==(const ::cuda::std::tuple<const exec_place&, const size_t&>& t) const
  {
    return stride_bytes == ::cuda::std::get<1>(t) && grid == ::cuda::std::get<0>(t);
  }

  exec_place grid;
  size_t stride_bytes;
  ::std::unique_ptr<localized_array> array;
  event_list prereqs;
};

/**
 * @brief A very simple allocation cache for slices in composite data places
 */
class composite_slice_cache
{
public:
  composite_slice_cache()                             = default;
  composite_slice_cache(const composite_slice_cache&) = delete;
  composite_slice_cache(composite_slice_cache&)       = delete;
  composite_slice_cache(composite_slice_cache&&)      = default;

  [[nodiscard]] event_list deinit()
  {
    event_list result;
    partition_fn_cache.each([&](auto& entry) {
      result.merge(mv(entry.prereqs));
      entry.prereqs.clear();
    });
    cute_partition_cache.each([&](auto& entry) {
      result.merge(mv(entry.prereqs));
      entry.prereqs.clear();
    });
    replicated_cache.each([&](auto& entry) {
      result.merge(mv(entry.prereqs));
      entry.prereqs.clear();
    });
    return result;
  }

  //! Take one replicated array for (grid, stride) from the cache, or build it
  [[nodiscard]] ::std::pair<::std::unique_ptr<localized_array>, event_list>
  get_replicated(const exec_place& grid, size_t stride_bytes)
  {
    auto entry         = replicated_cache.get(grid, stride_bytes);
    event_list prereqs = mv(entry->prereqs);
    return {mv(entry->array), mv(prereqs)};
  }

  //! Take every cached allocation from \p other (e.g. the cache of a popped
  //! nested context), gating any reuse on \p completion.
  //!
  //! The localized_array teardown unmaps VMM backing with synchronous driver
  //! calls that no event can defer, so a nested context's cached arrays must
  //! not be destroyed with it: they are handed over to the parent so their
  //! release happens once the parent has synchronized with the nested work
  //! (the parent's completion depends on the nested context's completion).
  //! \p completion should carry the nested body's completion events: the
  //! entries' own prereqs were already harvested by deinit() when the nested
  //! context was finalized, and a parent-level task reusing an entry must
  //! wait for the nested graph that last used it.
  void import_from(composite_slice_cache&& other, const event_list& completion)
  {
    other.partition_fn_cache.each([&](auto& entry) {
      entry.prereqs.merge(completion);
    });
    other.cute_partition_cache.each([&](auto& entry) {
      entry.prereqs.merge(completion);
    });
    other.replicated_cache.each([&](auto& entry) {
      entry.prereqs.merge(completion);
    });
    partition_fn_cache.import_from(mv(other.partition_fn_cache));
    cute_partition_cache.import_from(mv(other.cute_partition_cache));
    replicated_cache.import_from(mv(other.replicated_cache));
  }

  void put(const data_place& place,
           ::std::unique_ptr<localized_array> a,
           const event_list& prereqs,
           size_t total_size,
           size_t elem_size,
           dim4 data_dims)
  {
    const bool composite_or_replicated = place.is_composite() || place.is_replicated();
    EXPECT(composite_or_replicated);
    EXPECT(a.get());

    if (place.is_replicated())
    {
      const auto& grid = ::cuda::experimental::places::replicated_grid(place);
      auto entry =
        ::std::make_unique<cached_replicated_array>(grid, replicated_replica_stride(total_size * elem_size), mv(a));
      entry->prereqs.merge(prereqs);
      replicated_cache.put(mv(entry));
      return;
    }

    if (const auto* cute_place = as_cute_composite(place))
    {
      auto entry = ::std::make_unique<cached_cute_localized_array>(
        place.affine_exec_place(), cute_place->get_partition(), total_size, elem_size, data_dims, mv(a));
      entry->prereqs.merge(prereqs);
      cute_partition_cache.put(mv(entry));
      return;
    }

    auto entry = ::std::make_unique<cached_localized_array>(mv(a));
    entry->prereqs.merge(prereqs);
    partition_fn_cache.put(mv(entry));
  }

  template <typename F>
  ::std::pair<::std::unique_ptr<localized_array>, event_list>
  get(const data_place& place, F&& delinearize, size_t total_size, size_t elem_size, dim4 data_dims)
  {
    EXPECT(place.is_composite());

    if (const auto* cute_place = as_cute_composite(place))
    {
      const auto& partition = cute_place->get_partition();
      if (!(data_dims == partition.true_dims()))
      {
        throw ::std::invalid_argument("cute composite data_place: requested extents do not match the partition's true "
                                      "extents");
      }

      auto entry = cute_partition_cache.get(
        place.affine_exec_place(), partition, ::cuda::std::forward<F>(delinearize), total_size, elem_size, data_dims);
      event_list prereqs = mv(entry->prereqs);
      return {mv(entry->array), mv(prereqs)};
    }

    auto entry = partition_fn_cache.get(
      place.affine_exec_place(),
      place.get_partitioner(),
      ::cuda::std::forward<F>(delinearize),
      total_size,
      elem_size,
      data_dims);
    event_list prereqs = mv(entry->prereqs);
    return {mv(entry->array), mv(prereqs)};
  }

private:
  using cute_composite_place = ::cuda::experimental::places::data_place_cute_composite;

  static const cute_composite_place* as_cute_composite(const data_place& place)
  {
    return dynamic_cast<const cute_composite_place*>(place.get_impl().get());
  }

  linear_pool<cached_localized_array> partition_fn_cache;
  linear_pool<cached_cute_localized_array> cute_partition_cache;
  linear_pool<cached_replicated_array> replicated_cache;
};
} // end namespace cuda::experimental::stf::reserved
