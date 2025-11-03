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
 * @brief Implementation of constant_logical_data
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

#include <cuda/experimental/__stf/internal/logical_data.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <unordered_map>

namespace cuda::experimental::stf
{
/**
 * @brief A constant piece of data that relies on frozen logical data
 */
template <typename T>
class constant_logical_data
{
  class impl
  {
  public:
    // Initialize from an existing logical data
    template <typename ctx_t>
    impl(ctx_t& ctx, logical_data<T> orig_ld)
        : ld(mv(orig_ld))
        , frozen_ld(ctx.freeze(ld, access_mode::read))
        , stream(ctx.pick_stream())
    {
      // Freeze it and ensure it will be unfrozen automatically
      frozen_ld.set_automatic_unfreeze(true);

      // Cache a stream : we can use the same since we block on it so there
      // is no reason for taking different ones
    }

    // Return an instance of the data on the specified data place. This will block if necessary.
    T& get(const data_place& where = data_place::current_device())
    {
      auto it = cached.find(where);
      if (it != cached.end())
      {
        // There is a cached value, return it
        return it->second;
      }

      // We need to populate the cache

      // Insert the new value and get the iterator to the newly inserted item
      it = cached.emplace(where, frozen_ld.get(where, stream)).first;

      // Wait for the cache entry to be available
      cuda_safe_call(cudaStreamSynchronize(stream));

      return it->second;
    }

  private:
    logical_data<T> ld;
    frozen_logical_data<T> frozen_ld;

    // cached stream used to get instances
    cudaStream_t stream = {};

    // A cache of the instances, we can use them without further
    // synchronization once they are populated. If an instance is missing, we
    // will get it in a blocking manner.
    ::std::unordered_map<data_place, T, hash<data_place>> cached;
  };

public:
  template <typename ctx_t>
  constant_logical_data(ctx_t& ctx, logical_data<T> orig_ld)
      : pimpl(::std::make_shared<impl>(ctx, mv(orig_ld)))
  {}

  // So that we can have a constant data variable that is populated later
  constant_logical_data() = default;

  // Copy constructor
  constant_logical_data(const constant_logical_data& other) = default;

  // Move constructor
  constant_logical_data(constant_logical_data&& other) noexcept = default;

  // Copy assignment
  constant_logical_data& operator=(const constant_logical_data& other) = default;

  // Move assignment
  constant_logical_data& operator=(constant_logical_data&& other) noexcept = default;

  T& get(const data_place& where = data_place::current_device())
  {
    assert(pimpl);
    return pimpl->get(where);
  }

private:
  ::std::shared_ptr<impl> pimpl = nullptr;
};
} // end namespace cuda::experimental::stf
