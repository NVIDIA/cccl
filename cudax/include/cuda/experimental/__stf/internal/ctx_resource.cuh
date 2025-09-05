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
  ctx_resource(ctx_resource&&)                 = default;

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

/**
 * @brief Container for managing a set of resources associated with a context
 *
 * This class collects resources that need to be retained until a context has consumed them,
 * and provides efficient batched release mechanisms. Resources can be released either through
 * individual stream-based cleanup or batched host callbacks for better performance.
 *
 * The resource set is move-only to ensure proper resource ownership semantics.
 *
 * @par Usage Pattern:
 * @code
 * ctx_resource_set resources;
 * resources.add(::std::make_shared<my_resource>());
 * // ... context consumes resources ...
 * resources.release(stream);  // Release all resources asynchronously
 * @endcode
 */
class ctx_resource_set
{
  using resources_t = ::std::vector<::std::shared_ptr<ctx_resource>>;

public:
  ctx_resource_set() = default;

  // Non-copyable
  ctx_resource_set(const ctx_resource_set&)            = delete;
  ctx_resource_set& operator=(const ctx_resource_set&) = delete;

  // Move is fine
  ctx_resource_set(ctx_resource_set&&)            = default;
  ctx_resource_set& operator=(ctx_resource_set&&) = default;

  /**
   * @brief Add a resource to the set for managed release
   *
   * Stores a resource that needs to be retained until the context has consumed it.
   * The resource will be released when release() is called, either through individual
   * stream-based cleanup or batched host callbacks depending on the resource type.
   *
   * @param r Shared pointer to the resource to be managed
   *
   * @pre r must not be nullptr
   * @pre release() must not have been called on this resource set
   *
   * @par Example:
   * @code
   * auto my_res = std::make_shared<my_custom_resource>();
   * resource_set.add(my_res);
   * @endcode
   */
  void add(::std::shared_ptr<ctx_resource> r)
  {
    resources.push_back(mv(r));
  }

  /**
   * @brief Release all resources asynchronously with optimal batching
   *
   * Releases all managed resources using an optimized two-phase approach:
   * 1. Resources requiring stream-based cleanup are released individually
   * 2. Resources supporting host callbacks are batched into a single callback
   *
   * This batching strategy minimizes the overhead of host callbacks while ensuring
   * proper cleanup semantics for all resource types. After this call, the resource
   * set becomes empty and cannot be used for further resource additions.
   *
   * @param stream CUDA stream to use for resource cleanup and callbacks
   *
   * @pre Resources must not have been released already (resources not empty)
   * @post All resources are cleaned up and the resource set becomes empty
   * @post Further calls to add() or release() are invalid
   *
   * @par Performance Notes:
   * - Stream-dependent resources: Individual cleanup calls per resource
   * - Callback resources: Single batched host callback for all resources
   *
   * @par Example:
   * @code
   * ctx_resource_set resources;
   * resources.add(stream_resource);     // Released individually
   * resources.add(callback_resource);   // Released in batch
   * resources.release(cuda_stream);     // Efficient mixed cleanup
   * @endcode
   */
  void release(cudaStream_t stream)
  {
    // Separate resources into stream-dependent and callback-batched
    resources_t* callback_resources = nullptr;

    for (auto& r : resources)
    {
      if (r->can_release_in_callback())
      {
        if (!callback_resources)
        {
          callback_resources = new resources_t(1, mv(r));
        }
        else
        {
          callback_resources->push_back(mv(r));
        }
      }
      else
      {
        r->release(stream);
      }
    }
    resources_t().swap(resources); // Clear and deallocate the vector

    // Batch all callback resources into a single host callback for efficiency
    if (callback_resources)
    {
      // Add a single host callback using lambda that will release all callback resources
      auto release_lambda = [](cudaStream_t /*stream*/, cudaError_t /*status*/, void* userData) {
        auto* resources = static_cast<resources_t*>(userData);
        SCOPE(exit)
        {
          // Clean up the callback list itself
          delete resources;
        };

        // Release all callback resources
        for (auto& resource : *resources)
        {
          resource->release_in_callback();
        }
      };

      cuda_safe_call(cudaStreamAddCallback(stream, release_lambda, callback_resources, 0));
    }
  }

private:
  resources_t resources;
};

/**
 * @brief General-purpose implementation of `ctx_resource` for `new`-allocated pointers
 *
 * This class provides management for any pointer allocated with `new`
 * through the `ctx_resource` API. It performs cleanup by means of `delete`.
 *
 * @tparam Pointee The type of object being managed (will be deleted with `delete`)
 *
 * @par Usage Examples:
 * @code
 * // Managing temporary data structures
 * auto* temp_data = new MyDataStruct{...};
 * auto resource = std::make_shared<ctx_pointer_resource<MyDataStruct>>(temp_data);
 * resource_set.add(resource);
 *
 * // Managing callback arguments (original use case)
 * auto* callback_args = new CallbackArgs{...};
 * auto resource = std::make_shared<ctx_pointer_resource<CallbackArgs>>(callback_args);
 * resource_set.add(resource);
 *
 * // Managing arrays
 * auto* array = new int[100];
 * // Note: This class uses 'delete', not 'delete[]' - use appropriate wrapper for arrays
 * @endcode
 *
 * @note Despite the class name suggesting callback-specific usage, this is a
 *       general-purpose pointer management utility
 * @warning The managed pointer must have been allocated with `new` as it will be
 *          deallocated using `delete` (not `delete[]`)
 * @warning Do not manually delete the managed pointer - the resource handles cleanup
 */
template <typename Pointee, bool callback_releasable = true>
class ctx_pointer_resource : public ctx_resource
{
public:
  /**
   * @brief Construct a resource to manage the given pointer
   *
   * Takes ownership of a dynamically allocated pointer, ensuring it will be
   * automatically deleted when the resource is released.
   *
   * @param payload Pointer to be managed (must be allocated with `new`)
   */
  explicit ctx_pointer_resource(Pointee* payload)
      : payload_(payload)
  {}

  /**
   * @brief Indicates this resource can be safely released in a host callback
   *
   * @return Always returns true, enabling efficient batched cleanup of multiple
   *         pointer deletions in a single callback
   */
  bool can_release_in_callback() const override
  {
    return callback_releasable;
  }

  /**
   * @brief Release the managed resource by deleting the pointer
   *
   * Performs the actual memory deallocation using `delete`. This is called
   * during the batched cleanup phase to free the managed memory.
   */
  void release_in_callback() override
  {
    delete payload_;
  }

private:
  Pointee* payload_; //!< Pointer to the managed object that will be deleted during cleanup
};

} // end namespace cuda::experimental::stf
