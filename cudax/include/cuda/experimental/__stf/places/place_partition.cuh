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
 * This computes a vector of execution places which are the partition of the
 * input execution place divided with a specific partitioning granularity. One
 * may partition a place into devices, or green contexts, for example.
 *
 * An iterator is provided to dispatch computation over the subplaces.
 */
class place_partition
{
public:
  /** @brief Partition an execution place into a vector of subplaces */
  place_partition(async_resources_handle& handle, exec_place place, place_partition_scope scope)
  {
#if _CCCL_CTK_BELOW(12, 4)
    _CCCL_ASSERT(scope != place_partition_scope::green_context, "Green contexts unsupported.");
#endif // _CCCL_CTK_BELOW(12, 4)
    compute_subplaces(handle, place, scope);
  }

  /** @brief Partition a vector of execution places into a vector of subplaces */
  place_partition(
    async_resources_handle& handle, ::std::vector<::std::shared_ptr<exec_place>> places, place_partition_scope scope)
  {
    for (const auto& place : places)
    {
      compute_subplaces(handle, *place, scope);
    }
  }

  ~place_partition() = default;

  // To iterate over all subplaces
  using iterator = ::std::vector<exec_place>::iterator;
  iterator begin()
  {
    return sub_places.begin();
  }
  iterator end()
  {
    return sub_places.end();
    ;
  }

  /** @brief Number of subplaces in the partition */
  size_t size() const
  {
    return sub_places.size();
  }

  /** @brief Get the i-th place */
  exec_place& get(size_t i)
  {
    return sub_places[i];
  }

  /** @brief Get the i-th place (const reference) */
  const exec_place& get(size_t i) const
  {
    return sub_places[i];
  }

private:
  /** @brief Compute the subplaces of a place at the specified granularity (scope) into the sub_places vector */
  void compute_subplaces(async_resources_handle& handle, const exec_place& place, place_partition_scope scope)
  {
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

    if (place.is_grid() && scope == place_partition_scope::cuda_stream)
    {
      // Recursively partition grid into devices, then into streams
      for (auto& device_p : place_partition(handle, place, place_partition_scope::cuda_device))
      {
        auto device_p_places = place_partition(handle, device_p, place_partition_scope::cuda_stream).sub_places;
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
      for (auto& device_p : place_partition(handle, place, place_partition_scope::cuda_device))
      {
        auto device_p_places = place_partition(handle, device_p, place_partition_scope::green_context).sub_places;
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
#endif // _CCCL_CTK_AT_LEAST(12, 4)
    assert(!"Internal error: unreachable code.");
  }

  /** A vector with all subplaces (computed once in compute_subplaces) */
  ::std::vector<exec_place> sub_places;
};
} // end namespace cuda::experimental::stf
