//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Type-erased resources associated to contexts

#pragma once

#include <cuda/__cccl_config>

#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <functional>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental::stf
{
//! Generic container for a resource that needs to be retained until a context has consumed them
class ctx_resource
{
public:
  ctx_resource()          = default;
  virtual ~ctx_resource() = default;

  // Non-copyable
  ctx_resource(const ctx_resource&)            = delete;
  ctx_resource& operator=(const ctx_resource&) = delete;

  //! Release asynchronously (only called if can_release_in_callback() returns false)
  virtual void release(cudaStream_t)
  { /* Default implementation does nothing */
  }
  //! Returns true if this resource can be released in a host callback without using the stream
  //! Resources that return true will be batched together into a single callback to avoid
  //! the overhead of creating individual host callbacks for each resource release
  virtual bool can_release_in_callback() const
  {
    return false;
  }
  //! Release synchronously on the host (only called if can_release_in_callback() returns true)
  //! This will be called from within a batched host callback to minimize callback overhead
  virtual void release_in_callback()
  { /* Default implementation does nothing */
  }
};

class ctx_resource_set
{
public:
  ctx_resource_set() = default;

  // Non-copyable
  ctx_resource_set(const ctx_resource_set&)            = delete;
  ctx_resource_set& operator=(const ctx_resource_set&) = delete;

  // Move is fine
  ctx_resource_set(ctx_resource_set&&)            = default;
  ctx_resource_set& operator=(ctx_resource_set&&) = default;

  //! Store a resource until it is released
  void add(::std::shared_ptr<ctx_resource> r)
  {
    resources.push_back(mv(r));
  }

  //! Release all resources asynchronously
  void release(cudaStream_t stream)
  {
    _CCCL_ASSERT(!resources_released, "Resources have already been released on this context");

    // Separate resources into stream-dependent and callback-batched
    decltype(resources) callback_resources;

    for (auto& r : resources)
    {
      if (r->can_release_in_callback())
      {
        callback_resources.push_back(mv(r));
      }
      else
      {
        r->release(stream);
      }
    }
    resources.clear();

    // Batch all callback resources into a single host callback for efficiency
    if (!callback_resources.empty())
    {
      // Transfer ownership of callback resources to the callback
      auto* callback_list = new ::std::vector<::std::shared_ptr<ctx_resource>>(mv(callback_resources));

      // Add a single host callback using lambda that will release all callback resources
      auto release_lambda = [](cudaStream_t /*stream*/, cudaError_t /*status*/, void* userData) -> void {
        auto* resources = static_cast<::std::vector<::std::shared_ptr<ctx_resource>>*>(userData);

        // Release all callback resources
        for (auto& resource : *resources)
        {
          resource->release_in_callback();
        }

        // Clean up the callback list itself
        delete resources;
      };

      cuda_safe_call(cudaStreamAddCallback(stream, release_lambda, callback_list, 0));
    }

    // Mark as released to prevent double release
    resources_released = true;
  }

private:
  ::std::vector<::std::shared_ptr<ctx_resource>> resources;
  bool resources_released = false; // Safety flag to prevent double release
};
} // end namespace cuda::experimental::stf
