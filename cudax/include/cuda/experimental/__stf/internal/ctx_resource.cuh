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
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

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
  virtual void release(cudaStream_t) noexcept
  { /* Default implementation does nothing */
  }
  //! Returns true if this resource can be released in a host callback without using the stream
  //! Resources that return true will be batched together into a single callback to avoid
  //! the overhead of creating individual host callbacks for each resource release
  virtual bool can_release_in_callback() const noexcept
  {
    return false;
  }
  //! Release synchronously on the host (only called if can_release_in_callback() returns true)
  //! This will be called from within a batched host callback to minimize callback overhead
  virtual void release_in_callback() noexcept
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

    // Release stream-dependent resources and compact them out of `resources` by
    // pulling the last element into each vacated slot. A resource leaves
    // `resources` only after it has been released, so if release(stream) throws,
    // `resources` still holds the failing resource plus everything not yet
    // processed -- release() can be retried with nothing lost or double-released.
    for (size_t i = 0; i < resources.size();)
    {
      if (resources[i]->can_release_in_callback())
      {
        ++i;
        continue;
      }
      resources[i]->release(stream); // may throw -> resources[i] stays in place
      resources[i] = mv(resources.back());
      resources.pop_back();
    }

    if (!resources.empty())
    {
      // Transfer ownership of callback resources to the callback. Held in a
      // unique_ptr until the callback is successfully enqueued so a throw from
      // cudaStreamAddCallback does not leak the list.
      auto callback_list = ::std::make_unique<::std::vector<::std::shared_ptr<ctx_resource>>>(mv(resources));

      // Add a single host callback using lambda that will release all callback resources
      auto release_lambda = [](cudaStream_t /*stream*/, cudaError_t /*status*/, void* userData) -> void {
        auto* resources = static_cast<decltype(callback_list.get())>(userData);

        // Release all callback resources
        for (auto& resource : *resources)
        {
          // This is noexcept code
          resource->release_in_callback();
        }

        delete resources;
      };

      cuda_try<cudaStreamAddCallback>(stream, release_lambda, callback_list.get(), 0);
      callback_list.release();
    }

    // Mark as released to prevent double release
    resources_released = true;
  }

  //! Export all resources by moving them to a new ctx_resource_set
  //! The current set will be left empty after this operation
  ctx_resource_set export_resources()
  {
    _CCCL_ASSERT(!resources_released, "Cannot export resources that have already been released");

    ctx_resource_set exported;
    exported.resources          = mv(resources);
    exported.resources_released = resources_released;

    // Reset current state
    resources.clear();
    resources_released = false;

    return exported;
  }

  //! Import all resources from another ctx_resource_set
  //! The other set will be left empty after this operation
  void import_resources(ctx_resource_set&& other)
  {
    _CCCL_ASSERT(!resources_released, "Cannot import resources to a set that has already been released");
    _CCCL_ASSERT(!other.resources_released, "Cannot import resources that have already been released");

    // Move all resources from the other set to this set
    resources.insert(resources.end(),
                     ::std::make_move_iterator(other.resources.begin()),
                     ::std::make_move_iterator(other.resources.end()));

    // Clear the other set
    other.resources.clear();
  }

private:
  ::std::vector<::std::shared_ptr<ctx_resource>> resources;
  bool resources_released = false; // Safety flag to prevent double release
};
} // end namespace cuda::experimental::stf
