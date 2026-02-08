//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Base class for data_place extensions, enabling custom place types
 *
 * This extension mechanism allows custom data place types (like green contexts)
 * to be defined without modifying the core data_place class.
 * Extensions provide virtual methods for place-specific behavior like memory
 * allocation and string representation.
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

#include <cuda/std/__cccl/assert.h>

#include <cstddef>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::stf
{
// Forward declarations
class exec_place;

/**
 * @brief Base class for data_place extensions
 *
 * Custom data place types inherit from this class and override virtual methods
 * to provide place-specific behavior. This enables extensibility without
 * modifying the core data_place class.
 *
 * Example usage for a custom place type:
 * @code
 * class my_custom_extension : public data_place_extension {
 * public:
 *   exec_place get_affine_exec_place() const override { ... }
 *   int get_device_ordinal() const override { return my_device_id; }
 *   ::std::string to_string() const override { return "my_custom_place"; }
 *   size_t hash() const override { return std::hash<int>{}(my_device_id); }
 *   bool operator==(const data_place_extension& other) const override { ... }
 * };
 * @endcode
 */
class data_place_extension
{
public:
  virtual ~data_place_extension() = default;

  /**
   * @brief Get the affine execution place for this data place
   *
   * Returns the exec_place that should be used for computation on data
   * stored at this place. The exec_place may have its own virtual methods
   * (e.g., activate/deactivate) for execution-specific behavior.
   */
  virtual exec_place get_affine_exec_place() const = 0;

  /**
   * @brief Get the device ordinal for this place
   *
   * Returns the CUDA device ID associated with this place.
   * For host-only places, this should return -1.
   */
  virtual int get_device_ordinal() const = 0;

  /**
   * @brief Get a string representation of this place
   *
   * Used for debugging and logging purposes.
   */
  virtual ::std::string to_string() const = 0;

  /**
   * @brief Compute a hash value for this place
   *
   * Used for storing data_place in hash-based containers.
   */
  virtual size_t hash() const = 0;

  /**
   * @brief Check equality with another extension
   *
   * @param other The other extension to compare with
   * @return true if the extensions represent the same place
   */
  virtual bool operator==(const data_place_extension& other) const = 0;

  /**
   * @brief Compare ordering with another extension
   *
   * @param other The other extension to compare with
   * @return true if this extension is less than the other
   */
  virtual bool operator<(const data_place_extension& other) const = 0;

  /**
   * @brief Create a physical memory allocation for this place (VMM API)
   *
   * This method is used by localized arrays (composite_slice) to create physical
   * memory segments that are then mapped into a contiguous virtual address space.
   * Custom place types can override this method to provide specialized memory
   * allocation behavior.
   *
   * @note Managed memory is not supported by the VMM API.
   *
   * @param handle Output parameter for the allocation handle
   * @param size Size of the allocation in bytes
   * @return CUresult indicating success or failure
   *
   * @see allocate() for regular memory allocation
   */
  virtual CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const
  {
    int dev_ordinal = get_device_ordinal();

    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (dev_ordinal >= 0)
    {
      // Device memory
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id   = dev_ordinal;
    }
#if _CCCL_CTK_AT_LEAST(12, 2)
    else if (dev_ordinal == -1)
    {
      // Host memory (device ordinal -1)
      // CU_MEM_LOCATION_TYPE_HOST requires CUDA 12.2+
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
      prop.location.id   = 0;
    }
    else
    {
      // Managed memory (-2) is not supported by the VMM API
      _CCCL_ASSERT(false, "mem_create: managed memory is not supported by the VMM API");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv _CCCL_CTK_BELOW(12, 2) vvv
    else if (dev_ordinal == -1)
    {
      // Host VMM requires CU_MEM_LOCATION_TYPE_HOST which is only available in CUDA 12.2+
      _CCCL_ASSERT(false, "mem_create: host VMM requires CUDA 12.2+ (CU_MEM_LOCATION_TYPE_HOST not available)");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    else
    {
      // Managed memory (-2) is not supported by the VMM API
      _CCCL_ASSERT(false, "mem_create: managed memory is not supported by the VMM API");
      return CUDA_ERROR_NOT_SUPPORTED;
    }
#endif // _CCCL_CTK_AT_LEAST(12, 2)
    return cuMemCreate(handle, size, &prop, 0);
  }

  /**
   * @brief Allocate memory for this place (raw allocation)
   *
   * This is the low-level allocation interface. For stream-ordered allocations
   * (where allocation_is_stream_ordered() returns true), the allocation will
   * be ordered with respect to other operations on the stream. For immediate
   * allocations, the stream parameter is ignored.
   *
   * @param size Size of the allocation in bytes
   * @param stream CUDA stream for stream-ordered allocations (ignored for immediate allocations)
   * @return Pointer to allocated memory
   */
  virtual void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const = 0;

  /**
   * @brief Deallocate memory for this place (raw deallocation)
   *
   * @param ptr Pointer to memory to deallocate
   * @param size Size of the allocation
   * @param stream CUDA stream for stream-ordered deallocations (ignored for immediate deallocations)
   */
  virtual void deallocate(void* ptr, size_t size, cudaStream_t stream) const = 0;

  /**
   * @brief Returns true if allocation/deallocation is stream-ordered
   *
   * When this returns true, the allocation uses stream-ordered APIs like
   * cudaMallocAsync, and allocators should use stream_async_op to synchronize
   * prerequisites before allocation.
   *
   * When this returns false, the allocation is immediate (like cudaMallocHost)
   * and the stream parameter is ignored. Note that immediate deallocations
   * (e.g., cudaFree) may or may not introduce implicit synchronization.
   *
   * Default is true since most GPU-based extensions use cudaMallocAsync.
   */
  virtual bool allocation_is_stream_ordered() const
  {
    return true;
  }
};
} // end namespace cuda::experimental::stf
