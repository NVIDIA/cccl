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
 * @brief Abstract interface for data_place implementations
 *
 * This interface defines the contract that all data_place implementations must satisfy.
 * It enables a clean polymorphic design where host, managed, device, composite, and
 * extension-based places all implement a common interface.
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
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::stf
{
// Forward declarations
class exec_place;
class exec_place_grid;
class pos4;
class dim4;

//! Function type for computing executor placement from data coordinates
using get_executor_func_t = pos4 (*)(pos4, dim4, dim4);

/**
 * @brief Abstract interface for data_place implementations
 *
 * All data_place types (host, managed, device, composite, extensions) implement
 * this interface. The data_place class holds a shared_ptr to this interface
 * and delegates all operations to it.
 */
class data_place_interface
{
public:
  virtual ~data_place_interface() = default;

  /**
   * @brief Special device ordinal values for non-device places
   *
   * Returned by get_device_ordinal() for places that don't correspond
   * to a specific CUDA device.
   */
  enum ord : int
  {
    invalid     = ::std::numeric_limits<int>::min(),
    composite   = -5,
    device_auto = -4,
    affine      = -3,
    managed     = -2,
    host        = -1,
  };

  // === Type identification ===

  /**
   * @brief Check if this is the host (pinned memory) place
   */
  virtual bool is_host() const
  {
    return false;
  }

  /**
   * @brief Check if this is the managed memory place
   */
  virtual bool is_managed() const
  {
    return false;
  }

  /**
   * @brief Check if this is a specific device place
   */
  virtual bool is_device() const
  {
    return false;
  }

  /**
   * @brief Check if this is the invalid place
   */
  virtual bool is_invalid() const
  {
    return false;
  }

  /**
   * @brief Check if this is the affine place (uses exec_place's affine data place)
   */
  virtual bool is_affine() const
  {
    return false;
  }

  /**
   * @brief Check if this is the device_auto place (auto-select device)
   */
  virtual bool is_device_auto() const
  {
    return false;
  }

  /**
   * @brief Check if this is a composite place
   */
  virtual bool is_composite() const
  {
    return false;
  }

  /**
   * @brief Check if this is an extension-based place (green context, etc.)
   */
  virtual bool is_extension() const
  {
    return false;
  }

  /**
   * @brief Check if this is a concrete place (single, resolved allocation target)
   *
   * Returns true for host, managed, and device(ordinal). Returns false for
   * invalid, affine, device_auto, and composite, which must be resolved or
   * are not direct allocation targets. Extension places that support
   * allocation may override to return true.
   */
  virtual bool is_concrete() const
  {
    return false;
  }

  // === Core properties ===

  /**
   * @brief Get the device ordinal for this place
   *
   * Returns:
   * - >= 0 for specific CUDA devices
   * - data_place_ordinals::host (-1) for host
   * - data_place_ordinals::managed (-2) for managed
   * - data_place_ordinals::affine (-3) for affine
   * - data_place_ordinals::device_auto (-4) for device_auto
   * - data_place_ordinals::composite (-5) for composite
   * - data_place_ordinals::invalid for invalid
   */
  virtual int get_device_ordinal() const = 0;

  /**
   * @brief Get a string representation of this place
   */
  virtual ::std::string to_string() const = 0;

  /**
   * @brief Compute a hash value for this place
   */
  virtual size_t hash() const = 0;

  /**
   * @brief Three-way comparison with another place
   *
   * @return -1 if *this < other, 0 if *this == other, 1 if *this > other
   */
  virtual int cmp(const data_place_interface& other) const = 0;

  // === Memory allocation ===

  /**
   * @brief Allocate memory at this place
   *
   * @param size Size of the allocation in bytes
   * @param stream CUDA stream for stream-ordered allocations
   * @return Pointer to allocated memory
   * @throws std::runtime_error if allocation is not supported for this place type
   */
  virtual void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const = 0;

  /**
   * @brief Deallocate memory at this place
   *
   * @param ptr Pointer to memory to deallocate
   * @param size Size of the allocation
   * @param stream CUDA stream for stream-ordered deallocations
   */
  virtual void deallocate(void* ptr, size_t size, cudaStream_t stream) const = 0;

  /**
   * @brief Returns true if allocation/deallocation is stream-ordered
   */
  virtual bool allocation_is_stream_ordered() const = 0;

  /**
   * @brief Create a physical memory allocation for this place (VMM API)
   *
   * Default implementation returns CUDA_ERROR_NOT_SUPPORTED.
   * Subclasses that support VMM should override this.
   *
   * @param handle Output parameter for the allocation handle
   * @param size Size of the allocation in bytes
   * @return CUresult indicating success or failure
   */
  virtual CUresult mem_create(CUmemGenericAllocationHandle*, size_t) const
  {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  // === Extension support ===

  /**
   * @brief Get the implementation for the affine exec_place (for extensions)
   *
   * Extensions override this to provide their affine exec_place implementation.
   * Returns nullptr by default (non-extensions).
   * The returned shared_ptr should be castable to shared_ptr<exec_place::impl>.
   */
  virtual ::std::shared_ptr<void> get_affine_exec_impl() const
  {
    throw ::std::logic_error("get_affine_exec_impl() called on non-composite data_place");
  }

  // === Composite-specific (throw by default) ===

  /**
   * @brief Get the grid for composite places
   * @throws std::logic_error if not a composite place
   */
  virtual const exec_place_grid& get_grid() const
  {
    throw ::std::logic_error("get_grid() called on non-composite data_place");
  }

  /**
   * @brief Get the partitioner function for composite places
   * @throws std::logic_error if not a composite place
   */
  virtual const get_executor_func_t& get_partitioner() const
  {
    throw ::std::logic_error("get_partitioner() called on non-composite data_place");
  }
};
} // end namespace cuda::experimental::stf
