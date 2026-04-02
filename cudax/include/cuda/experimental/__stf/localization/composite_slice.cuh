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

#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__stf/internal/async_prereq.cuh>

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
    cache.each([&](auto& entry) {
      result.merge(mv(entry.prereqs));
      entry.prereqs.clear();
    });
    return result;
  }

  void put(::std::unique_ptr<localized_array> a, const event_list& prereqs)
  {
    EXPECT(a.get());
    auto entry = ::std::make_unique<cached_localized_array>(mv(a));
    entry->prereqs.merge(prereqs);
    cache.put(mv(entry));
  }

  template <typename F>
  ::std::pair<::std::unique_ptr<localized_array>, event_list> get(
    const data_place& place, partition_fn_t mapper, F&& delinearize, size_t total_size, size_t elem_size, dim4 data_dims)
  {
    EXPECT(place.is_composite());
    auto entry =
      cache.get(place.affine_exec_place(), mapper, ::std::forward<F>(delinearize), total_size, elem_size, data_dims);
    event_list prereqs = mv(entry->prereqs);
    return {mv(entry->array), mv(prereqs)};
  }

private:
  linear_pool<cached_localized_array> cache;
};
} // end namespace cuda::experimental::stf::reserved
