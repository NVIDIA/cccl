//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Facilities to manipulate subset of places
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

#include <cuda/experimental/__stf/internal/exec_affinity.cuh>
#include <cuda/experimental/__stf/places/exec/cuda_stream.cuh>
#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/__stf/places/places.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Defines a partitioning granularity
 *
 * This should be used in combination with `place_partition`
 */
enum class place_partition_scope
{
  cuda_device,
  green_context,
  cuda_stream,
};

/**
 * @brief Convert a place_partition_scope value to a string (for debugging purpose)
 * @param scope The partitioning granularity to convert
 * @return A string representation of `scope` (e.g. "cuda_device", "green_context", "cuda_stream")
 */
inline ::std::string place_partition_scope_to_string(place_partition_scope scope)
{
  switch (scope)
  {
    case place_partition_scope::cuda_device:
      return "cuda_device";
    case place_partition_scope::green_context:
      return "green_context";
    case place_partition_scope::cuda_stream:
      return "cuda_stream";
  }

  abort();
  return "unknown";
}

// TODO method to get the scope of an exec place

/**
 * @brief Get subsets of an execution place.
 *
 * Computes a vector of execution places that partition the input place at a
 * given granularity (see `place_partition_scope`). For example, a grid place
 * can be partitioned into devices, or into green contexts, or into CUDA streams.
 *
 * Use the constructors that take `async_resources_handle&` when partitioning at
 * `cuda_stream` or `green_context` scope (stream and green-context resources
 * are obtained from the handle). The constructors without a handle support only
 * `cuda_device` scope. Green context scope requires CUDA 12.4 or later.
 *
 * Iteration over subplaces is provided via `begin()` / `end()`; `as_grid()` builds
 * an `exec_place_grid` from the subplaces.
 */
class place_partition
{
public:
  /** @brief Partition an execution place into a vector of subplaces (with async resource handle).
   * @param place The execution place to partition (e.g. grid or device)
   * @param handle Handle used to obtain stream or green-context resources when scope is cuda_stream or green_context
   * @param scope Partitioning granularity (cuda_device, green_context, or cuda_stream)
   */
  place_partition(exec_place place, async_resources_handle& handle, place_partition_scope scope)
  {
#if _CCCL_CTK_BELOW(12, 4)
    _CCCL_ASSERT(scope != place_partition_scope::green_context, "Green contexts unsupported.");
#endif // _CCCL_CTK_BELOW(12, 4)
    compute_subplaces(handle, place, scope);
  }

  /** @brief Partition an execution place into a vector of subplaces (no async handle).
   * Only `cuda_device` scope is supported; green_context and cuda_stream require a handle.
   * @param place The execution place to partition
   * @param scope Partitioning granularity (must be cuda_device when no handle is provided)
   */
  place_partition(exec_place place, place_partition_scope scope)
  {
#if _CCCL_CTK_BELOW(12, 4)
    _CCCL_ASSERT(scope != place_partition_scope::green_context, "Green contexts need an async resource handle.");
#endif // _CCCL_CTK_BELOW(12, 4)
    compute_subplaces_no_handle(place, scope);
  }

  /** @brief Partition a vector of execution places into a single vector of subplaces (with async handle).
   * @param handle Handle for stream or green-context resources when scope is cuda_stream or green_context
   * @param places Input execution places to partition
   * @param scope Partitioning granularity
   */
  place_partition(async_resources_handle& handle,
                  const ::std::vector<::std::shared_ptr<exec_place>>& places,
                  place_partition_scope scope)
  {
    for (const auto& place : places)
    {
      compute_subplaces(handle, *place, scope);
    }
  }

  /** @brief Partition a grid of execution places into a single vector of subplaces (with async handle).
   * @param handle Handle for stream or green-context resources when scope is cuda_stream or green_context
   * @param grid Input execution place grid to partition
   * @param scope Partitioning granularity
   */
  place_partition(async_resources_handle& handle, const exec_place_grid& grid, place_partition_scope scope)
  {
    ::std::vector<::std::shared_ptr<exec_place>> places;
    const auto& grid_places = grid.get_places();
    places.reserve(grid_places.size());
    for (const auto& ep : grid_places)
    {
      places.push_back(::std::make_shared<exec_place>(ep));
    }
    for (const auto& place : places)
    {
      compute_subplaces(handle, *place, scope);
    }
  }

  /** @brief Partition a vector of execution places into a single vector of subplaces (no async handle).
   * Only cuda_device scope is supported.
   * @param places Input execution places to partition
   * @param scope Partitioning granularity (must be cuda_device)
   */
  place_partition(const ::std::vector<::std::shared_ptr<exec_place>>& places, place_partition_scope scope)
  {
    for (const auto& place : places)
    {
      compute_subplaces_no_handle(*place, scope);
    }
  }

  ~place_partition() = default;

  /** Iteration over subplaces. */
  using iterator       = ::std::vector<exec_place>::iterator;
  using const_iterator = ::std::vector<exec_place>::const_iterator;

  /** @brief Iterator to the first subplace. @return Begin iterator. */
  iterator begin()
  {
    return sub_places.begin();
  }
  /** @brief Past-the-end iterator for subplaces. @return End iterator. */
  iterator end()
  {
    return sub_places.end();
  }
  /** @brief Const iterator to the first subplace. @return Begin const iterator. */
  const_iterator begin() const
  {
    return sub_places.begin();
  }
  /** @brief Past-the-end const iterator. @return End const iterator. */
  const_iterator end() const
  {
    return sub_places.end();
  }

  /** @brief Number of subplaces in the partition. @return Size of the partition. */
  size_t size() const
  {
    return sub_places.size();
  }

  /** @brief Get the i-th subplace (mutable).
   * @param i Index in [0, size()).
   * @return Reference to the i-th exec_place.
   */
  exec_place& get(size_t i)
  {
    return sub_places[i];
  }

  /** @brief Get the i-th subplace (const).
   * @param i Index in [0, size()).
   * @return Const reference to the i-th exec_place.
   */
  const exec_place& get(size_t i) const
  {
    return sub_places[i];
  }

  /** @brief Build an exec_place_grid from the subplaces.
   * @return A grid view of the partitioned execution places.
   */
  exec_place_grid as_grid() const
  {
    return make_grid(sub_places);
  }

private:
  /** @brief Compute the subplaces of a place at the specified granularity (scope) into the sub_places vector */
  void compute_subplaces(async_resources_handle& handle, const exec_place& place, place_partition_scope scope)
  {
    if (place.is_grid() && scope == place_partition_scope::cuda_stream)
    {
      // Recursively partition grid into devices, then into streams
      for (auto& device_p : place_partition(place, handle, place_partition_scope::cuda_device))
      {
        auto device_p_places = place_partition(device_p, handle, place_partition_scope::cuda_stream).sub_places;
        sub_places.insert(sub_places.end(), device_p_places.begin(), device_p_places.end());
      }
      return;
    }

    if (place.is_device() && scope == place_partition_scope::cuda_stream)
    {
      auto& pool = place.get_stream_pool(handle, true);
      for (size_t i = 0; i < pool.size(); i++)
      {
        // As a side effect, this will populate the pool
        decorated_stream dstream = pool.next();
        sub_places.push_back(exec_place::cuda_stream(dstream));
      }
      return;
    }

// Green contexts are only supported since CUDA 12.4
#if _CCCL_CTK_AT_LEAST(12, 4)
    if (place.is_grid() && scope == place_partition_scope::green_context)
    {
      // Recursively partition grid into devices, then into green contexts
      for (auto& device_p : place_partition(place, handle, place_partition_scope::cuda_device))
      {
        auto device_p_places = place_partition(device_p, handle, place_partition_scope::green_context).sub_places;
        sub_places.insert(sub_places.end(), device_p_places.begin(), device_p_places.end());
      }
      return;
    }

    if (place.is_device() && scope == place_partition_scope::green_context)
    {
      // Find the device associated to the place, and get the green context helper
      int dev_id = device_ordinal(place.affine_data_place());

      // 8 SMs per green context is a granularity that should work on any arch.
      const char* env = getenv("CUDASTF_GREEN_CONTEXT_SIZE");
      int sm_cnt      = env ? atoi(env) : 8;

      auto h = handle.get_gc_helper(dev_id, sm_cnt);

      // Get views of green context out of the helper to create execution places
      size_t cnt = h->get_count();
      for (size_t i = 0; i < cnt; i++)
      {
        sub_places.push_back(exec_place::green_ctx(h->get_view(i)));
      }
      return;
    }
#endif

    // If the scope requires no handle
    compute_subplaces_no_handle(place, scope);
  }

  /** @brief Compute the subplaces of a place at the specified granularity (scope) into the sub_places vector */
  void compute_subplaces_no_handle(const exec_place& place, place_partition_scope scope)
  {
#if _CCCL_CTK_BELOW(12, 4)
    _CCCL_ASSERT(scope != place_partition_scope::green_context, "Green contexts scope need an async resource handle.");
#endif // _CCCL_CTK_BELOW(12, 4)
    _CCCL_ASSERT(scope != place_partition_scope::cuda_stream, "CUDA stream scope needs an async resource handle.");

    if (place.is_grid() && scope == place_partition_scope::cuda_device)
    {
      exec_place_grid g = place.as_grid();
      // Copy the vector of places
      sub_places = g.get_places();
      return;
    }

    if (place.is_device() && scope == place_partition_scope::cuda_device)
    {
      sub_places.push_back(place);
      return;
    }

    assert(!"Internal error: unreachable code.");
  }

  /** A vector with all subplaces (computed once in compute_subplaces) */
  ::std::vector<exec_place> sub_places;
};

// Deferred implementation because we need place_partition
template <typename... Args>
auto exec_place::partition_by_scope(Args&&... args)
{
  return place_partition(*this, ::std::forward<Args>(args)...).as_grid();
}
} // end namespace cuda::experimental::stf
