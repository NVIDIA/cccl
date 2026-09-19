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
 * custom places (e.g. green contexts) all implement a common interface.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/limits>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__exception/exception_macros.h>

#include <cuda/experimental/__stf/utility/dimensions.cuh>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::places
{
using ::cuda::experimental::stf::dim4;
using ::cuda::experimental::stf::pos4;

// Forward declarations
class exec_place;

//! Function type for computing executor placement from data coordinates.
//! Uses an out-pointer convention so the signature is trivially representable
//! in FFI frameworks (ctypes, cffi, Rust) that cannot return C structs.
using partition_fn_t = void (*)(pos4* result, pos4 data_coords, dim4 data_dims, dim4 grid_dims);

/**
 * @brief The partitioner of a composite data place: how it maps data
 * coordinates to a grid position, plus the identity that distinguishes it
 * from other partitioners.
 *
 * A plain partition_fn_t converts implicitly and is its own identity, so
 * existing callers are unaffected. The second form exists for a mapper that
 * cannot be *called* as a partition_fn_t - one reached through a foreign ABI,
 * whose coordinate structs are its own types, or a stateful partitioner. Such
 * a caller wraps the call in an adapter and supplies a separate identity.
 *
 * Identity is what makes two composite places compare (and order) equal, and
 * what lets the localized_array cache reuse a mapping, so it must be stable
 * for a given partitioner. Note the identity is only ever compared, never
 * called: reinterpret_cast between function pointer types is well defined as
 * long as the result is not invoked, which is exactly how a foreign-ABI
 * callback can serve as one.
 *
 * LIFETIME CONTRACT (foreign-ABI callers): because caches key on the identity
 * and may outlive the data place, the identity must stay unique for as long
 * as any mapping derived from it can be reused. A trampoline freed and
 * reallocated at the same address for a DIFFERENT mapper would alias the old
 * cache entry and silently serve the old mapper's placement. Bindings must
 * therefore keep their trampoline (and thus its address) alive for the
 * lifetime of any context that may cache placements from it. The Python
 * bindings currently pin the trampoline for the *data place's* lifetime,
 * which narrows but does not close the window (a cache can outlive the
 * place); making the cache robust to identity reuse (generation-tagged
 * identities, or invalidation on mapper destruction) is follow-up work.
 */
class partition_mapper
{
public:
  using call_type = ::std::function<void(pos4* result, pos4 data_coords, dim4 data_dims, dim4 grid_dims)>;

  partition_mapper() = default;

  //! A raw partition function is its own identity
  /*implicit*/ partition_mapper(partition_fn_t fn)
      : call_(fn)
      , identity_(fn)
  {}

  //! A mapper that is not callable as a partition_fn_t, with an explicit identity
  partition_mapper(call_type call, partition_fn_t identity)
      : call_(::std::move(call))
      , identity_(identity)
  {}

  void operator()(pos4* result, pos4 data_coords, dim4 data_dims, dim4 grid_dims) const
  {
    call_(result, data_coords, data_dims, grid_dims);
  }

  [[nodiscard]] explicit operator bool() const noexcept
  {
    return static_cast<bool>(call_);
  }

  [[nodiscard]] friend bool operator==(const partition_mapper& lhs, const partition_mapper& rhs) noexcept
  {
    return lhs.identity_ == rhs.identity_;
  }

  [[nodiscard]] friend bool operator!=(const partition_mapper& lhs, const partition_mapper& rhs) noexcept
  {
    return !(lhs == rhs);
  }

  [[nodiscard]] friend bool operator<(const partition_mapper& lhs, const partition_mapper& rhs) noexcept
  {
    return ::std::less<partition_fn_t>{}(lhs.identity_, rhs.identity_);
  }

private:
  call_type call_;
  partition_fn_t identity_ = nullptr;
};

/**
 * @brief Abstract interface for data_place implementations
 *
 * All data_place types (host, managed, device, composite, future places) implement
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
    invalid     = ::cuda::std::numeric_limits<int>::min(),
    composite   = -5,
    device_auto = -4,
    affine      = -3,
    managed     = -2,
    host        = -1,
  };

  // === Core properties ===

  /**
   * @brief Whether this place is fully resolved and ready for allocation
   *
   * Returns true for places that represent a concrete memory target:
   * host, managed, device(N), composite, green_ctx, etc.
   * Returns false for abstract/deferred places that need further
   * resolution: invalid, affine, device_auto.
   */
  virtual bool is_resolved() const = 0;

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
   * This is a standalone entry point: callers are not required to activate
   * this place or make any particular device current beforehand, so
   * implementations must not assume the calling thread's current device (or
   * context) matches this place. An implementation that needs to switch must
   * restore the caller's current device before returning.
   *
   * @param size Size of the allocation in bytes
   * @param stream CUDA stream for stream-ordered allocations
   * @return Pointer to allocated memory
   * @throws std::runtime_error if allocation is not supported for this place type
   */
  virtual void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const = 0;

  /**
   * @brief Allocate memory at this place for a tensor with the given extents
   *
   * The default implementation ignores the tensor geometry and forwards to the
   * byte-count allocate(); places whose physical placement depends on the
   * geometry (composite places, whose partitioner maps element coordinates to
   * places) override it with the real implementation.
   *
   * Extents follow the dimension-0-fastest linearization convention of
   * dim4::get_index() (the STF slice convention). Row-major callers should
   * present reversed extents (and a coordinate-reversing partitioner).
   *
   * The standalone contract of allocate() applies here as well: the caller's
   * current device is unspecified on entry and must be left unchanged on
   * return.
   *
   * @param data_dims Extents of the tensor
   * @param elemsize Size of one element in bytes
   * @param stream CUDA stream for stream-ordered allocations
   * @return Pointer to allocated memory
   */
  virtual void* allocate_nd(dim4 data_dims, size_t elemsize, cudaStream_t stream) const
  {
    return allocate(static_cast<::std::ptrdiff_t>(data_dims.size() * elemsize), stream);
  }

  /**
   * @brief Deallocate memory at this place
   *
   * Same standalone contract as allocate(): the caller's current device is
   * unspecified on entry and must be left unchanged on return.
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
   * Same standalone contract as allocate(): the caller's current device is
   * unspecified on entry and must be left unchanged on return. Placement must
   * come from the explicit allocation properties (CUmemAllocationProp), not
   * from the current device.
   *
   * @param handle Output parameter for the allocation handle
   * @param size Size of the allocation in bytes
   * @return CUresult indicating success or failure
   */
  virtual CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const
  {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  /**
   * @brief Get the implementation for the affine exec_place (for custom place types)
   *
   * Custom data_place implementations (e.g. green contexts) override this to
   * provide their own affine exec_place. Returns nullptr by default, which
   * causes data_place::affine_exec_place() to fall through to the error path.
   * The returned shared_ptr should be castable to shared_ptr<exec_place::impl>.
   */
  virtual ::std::shared_ptr<void> get_affine_exec_impl() const
  {
    return nullptr;
  }

  // === Composite-specific (throw by default) ===

  /**
   * @brief Whether this place is a composite place (data distributed over a
   * grid of places by a partitioner)
   */
  virtual bool is_composite() const
  {
    return false;
  }

  //! Whether this is a replicated data place (one copy per grid member)
  virtual bool is_replicated() const noexcept
  {
    return false;
  }

  //! Number of data instances a dependency at this place resolves to: 1 for
  //! ordinary and composite places, one per grid member for a replicated
  //! place (see data_place::member for the r-th instance's place)
  virtual size_t instance_count() const
  {
    return 1;
  }

  /**
   * @brief Get the partitioner function for composite places
   * @throws std::logic_error if not a composite place
   */
  virtual const partition_mapper& get_partitioner() const
  {
    _CCCL_THROW(::std::logic_error, "get_partitioner() called on non-composite data_place");
  }
};
} // end namespace cuda::experimental::places
