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

#include <cuda/experimental/__places/cute_partition.cuh>
#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/internal/stf_places_extended_exports.cuh>
#include <cuda/experimental/__stf/internal/stf_places_into_stf_core.cuh>

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
      if (*e == ::std::tuple<const P&...>(p...))
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

    return ::std::make_unique<T>(::std::forward<P>(p)...);
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

  template <typename... Args, typename = ::std::enable_if_t<::std::is_constructible_v<localized_array, Args...>>>
  explicit cached_localized_array(Args&&... args)
      : array(::std::make_unique<localized_array>(::std::forward<Args>(args)...))
  {}

  template <typename... P>
  bool operator==(::std::tuple<P&...> t) const
  {
    return *array == t;
  }

  ::std::unique_ptr<localized_array> array;
  event_list prereqs;
};

/**
 * @brief Cached localized array whose placement is described by a
 *        cute_partition value.
 *
 * Unlike partition_fn_t, a cute_partition is stateful. Keep its value with
 * the cached allocation so independently constructed equivalent composite
 * places can reuse the same VMM mapping.
 */
struct cached_cute_localized_array
{
  template <typename F>
  explicit cached_cute_localized_array(
    exec_place grid_,
    ::cuda::experimental::places::cute_partition partition_,
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
      [partition = this->partition, delinearize = ::std::forward<F>(delinearize)](size_t ind) {
        return partition.owner(delinearize(ind));
      });
    array = ::std::make_unique<localized_array>(grid, owner_of, total_size, elem_size, data_dims);
  }

  explicit cached_cute_localized_array(
    exec_place grid_,
    ::cuda::experimental::places::cute_partition partition_,
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
  bool operator==(::std::tuple<P&...> t) const
  {
    // tuple arguments:
    // 0: grid, 1: partition, 2: delinearize function, 3: total size,
    // 4: element size, 5: data dimensions
    return grid == ::std::get<0>(t) && partition == ::std::get<1>(t)
        && total_size_bytes == ::std::get<3>(t) * ::std::get<4>(t) && elemsize == ::std::get<4>(t)
        && data_dims == ::std::get<5>(t);
  }

  exec_place grid;
  ::cuda::experimental::places::cute_partition partition;
  size_t total_size_bytes;
  dim4 data_dims;
  size_t elemsize;
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
    return result;
  }

  void put(const data_place& place,
           ::std::unique_ptr<localized_array> a,
           const event_list& prereqs,
           size_t total_size,
           size_t elem_size,
           dim4 data_dims)
  {
    EXPECT(place.is_composite());
    EXPECT(a.get());

    if (const auto* cute_place =
          dynamic_cast<const ::cuda::experimental::places::data_place_cute_composite*>(place.get_impl().get()))
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

    if (const auto* cute_place =
          dynamic_cast<const ::cuda::experimental::places::data_place_cute_composite*>(place.get_impl().get()))
    {
      auto entry = cute_partition_cache.get(
        place.affine_exec_place(),
        cute_place->get_partition(),
        ::std::forward<F>(delinearize),
        total_size,
        elem_size,
        data_dims);
      event_list prereqs = mv(entry->prereqs);
      return {mv(entry->array), mv(prereqs)};
    }

    auto entry = partition_fn_cache.get(
      place.affine_exec_place(),
      place.get_partitioner(),
      ::std::forward<F>(delinearize),
      total_size,
      elem_size,
      data_dims);
    event_list prereqs = mv(entry->prereqs);
    return {mv(entry->array), mv(prereqs)};
  }

private:
  linear_pool<cached_localized_array> partition_fn_cache;
  linear_pool<cached_cute_localized_array> cute_partition_cache;
};
} // end namespace cuda::experimental::stf::reserved
