//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H
#define _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__device/attributes.h>
#  include <cuda/__device/device_ref.h>
#  include <cuda/__memory_pool/attributes.h>
#  include <cuda/__memory_resource/any_resource.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/internal_streams.h>
#  include <cuda/__stream/stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/cstddef>

#  if _CCCL_HOSTED()
#    include <vector>
#  endif // _CCCL_HOSTED()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
// clang complains about missing braces in CUmemLocation constructor but GCC complains if we add them

//! @brief Enable access to this memory pool from the supplied devices
//!
//! Device on which this pool resides can be included in the span.
//!
//! @param __pool The memory pool to set access for
//! @param __devices A span of `device_ref`s listing devices to enable access
//! for
//! @param __flags The access flags to set
//! @throws cuda_error if ``cudaMemPoolSetAccess`` fails.
inline void
__mempool_set_access(::CUmemoryPool __pool, ::cuda::std::span<const device_ref> __devices, ::CUmemAccess_flags __flags)
{
  ::std::vector<::CUmemAccessDesc> __descs;
  __descs.reserve(__devices.size());
  for (const auto& __dev : __devices)
  {
    __descs.push_back({::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __dev.get()}, __flags});
  }
  ::cuda::__driver::__mempoolSetAccess(__pool, __descs.data(), __descs.size());
}

//! @brief Query if memory from a pool is accessible by the supplied device
//!
//! @param __pool The memory pool to query access for
//! @param __dev The device to query access for
//! @returns true if the memory pool is accessible from the device
[[nodiscard]] inline bool __mempool_get_access(::cudaMemPool_t __pool, device_ref __dev)
{
  ::CUmemAccess_flags __result;
  ::CUmemLocation __loc;
  __loc.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
  __loc.id   = __dev.get();
  __result   = ::cuda::__driver::__mempoolGetAccess(__pool, &__loc);
  return __result == ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}

//! @brief \c memory_pool_properties is a type that can controls memory pool to
//! control the creation options. Compared to attributes, properties can not be
//! set after the pool is created.
struct memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = ::cuda::std::numeric_limits<size_t>::max();
  cudaMemAllocationHandleType allocation_handle_type = ::cudaMemAllocationHandleType::cudaMemHandleTypeNone;
  size_t max_pool_size                               = 0;
};

//! @brief  Creates the CUDA memory pool from the passed in arguments.
//! @throws cuda_error If the creation of the CUDA memory pool failed.
//! @returns The created CUDA memory pool.
[[nodiscard]] static cudaMemPool_t __create_cuda_mempool(
  memory_pool_properties __properties, ::CUmemLocation __location, CUmemAllocationType __allocation_type)
{
  ::CUmemPoolProps __pool_properties{};
  __pool_properties.allocType   = __allocation_type;
  __pool_properties.handleTypes = ::CUmemAllocationHandleType(__properties.allocation_handle_type);
  __pool_properties.location    = __location;

#  if _CCCL_CTK_AT_LEAST(12, 2)
  if (__properties.max_pool_size != 0)
  {
#    if _CCCL_CTK_AT_LEAST(13, 0)
    if (__allocation_type == ::CU_MEM_ALLOCATION_TYPE_MANAGED)
    {
      _CCCL_THROW(::std::invalid_argument, "Max pool size is not supported for managed memory pools");
    }
#    endif // _CCCL_CTK_AT_LEAST(13, 0)
    if (__properties.initial_pool_size > __properties.max_pool_size)
    {
      _CCCL_THROW(::std::invalid_argument, "Initial pool size must be less than the max pool size");
    }
  }
  __pool_properties.maxSize = __properties.max_pool_size;
#  else
  if (__properties.max_pool_size != 0)
  {
    _CCCL_THROW(::std::invalid_argument, "Max pool size is not supported on this CUDA version");
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 2)

  if (__properties.initial_pool_size > __properties.release_threshold)
  {
    _CCCL_THROW(::std::invalid_argument, "Initial pool size must be less than the release threshold");
  }

  ::CUmemoryPool __cuda_pool_handle{};
  ::cudaError_t __error = ::cuda::__driver::__mempoolCreateNoThrow(&__cuda_pool_handle, &__pool_properties);
  if (__error != ::cudaSuccess)
  {
    auto __device = __location.type == ::CU_MEM_LOCATION_TYPE_DEVICE ? __location.id : 0;
    // Mempool creation failed, lets try to figure out why
    ::cuda::__verify_device_supports_stream_ordered_allocations(__location, __allocation_type);
    ::cuda::__verify_device_supports_export_handle_type(__device, __properties.allocation_handle_type, __location);

    // Could not find the reason, throw a generic error
    _CCCL_THROW(::cuda::cuda_error, __error, "Failed to create a memory pool");
  }

  ::cuda::__driver::__mempoolSetAttribute(
    __cuda_pool_handle, ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &__properties.release_threshold);

  // allocate the requested initial size to prime the pool.
  // We need to use a new stream so we do not wait on other work
  if (__properties.initial_pool_size != 0)
  {
    ::CUdeviceptr __ptr = ::cuda::__driver::__mallocFromPoolAsync(
      __properties.initial_pool_size, __cuda_pool_handle, __cccl_allocation_stream().get());
    if (::cuda::__driver::__freeAsyncNoThrow(__ptr, __cccl_allocation_stream().get()) != ::cudaSuccess)
    {
      _CCCL_THROW(::cuda::cuda_error, ::cudaErrorMemoryAllocation, "Failed to allocate initial pool size");
    }
  }
  return __cuda_pool_handle;
}

class __memory_pool_base
{
protected:
  ::cudaMemPool_t __pool_;

  //! @brief Checks whether the passed in alignment is valid.
  //! @param __alignment the alignment to check.
  //! @returns true if \p __alignment is valid.
  [[nodiscard]] _CCCL_HOST_API static constexpr bool __is_valid_alignment(const size_t __alignment) noexcept
  {
    return __alignment <= ::cuda::mr::default_cuda_malloc_alignment
        && (::cuda::mr::default_cuda_malloc_alignment % __alignment == 0);
  }

public:
  __memory_pool_base(::cuda::std::nullptr_t) = delete;

  //! @brief  Constructs the __memory_pool_base from a \c cudaMemPool_t.
  //! @param __pool The \c cudaMemPool_t used to allocate memory.
  _CCCL_HOST_API explicit __memory_pool_base(::cudaMemPool_t __pool) noexcept
      : __pool_(__pool)
  {}

  //! @brief Allocate device memory of size at least \p __bytes via
  //! cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the CUDA API call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void*
  allocate_sync(const size_t __bytes, const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid alignment passed to __memory_pool_base::allocate_sync.");
    }

    ::CUdeviceptr __ptr = ::cuda::__driver::__mallocFromPoolAsync(__bytes, __pool_, __cccl_allocation_stream().get());
    __cccl_allocation_stream().sync();
    return reinterpret_cast<void*>(__ptr); // NOLINT(performance-no-int-to-ptr)
  }

  //! @brief deallocate_sync memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate_sync`.
  //! @param __bytes  The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call
  //! that returned \p __ptr.
  //! @note The pointer passed to `deallocate_sync` must not be in use in a
  //! stream. It is the caller's responsibility to properly synchronize all
  //! relevant streams before calling `deallocate_sync`.
  _CCCL_HOST_API void deallocate_sync(
    void* __ptr,
    const size_t,
    [[maybe_unused]] const size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to __memory_pool_base::deallocate_sync.");
    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeAsyncNoThrow,
      "deallocate failed",
      reinterpret_cast<::CUdeviceptr>(__ptr),
      __cccl_allocation_stream().get());
  }

  //! @brief Allocate device memory of size at least \p __bytes via
  //! `cudaMallocFromPoolAsync`.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __alignment The requested alignment of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws std::invalid_argument In case of invalid alignment.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void*
  allocate(const ::cuda::stream_ref __stream, const size_t __bytes, const size_t __alignment)
  {
    if (!__is_valid_alignment(__alignment))
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid alignment passed to __memory_pool_base::allocate.");
    }

    return allocate(__stream, __bytes);
  }

  //! @brief Allocate device memory of size at least \p __bytes via
  //! cudaMallocFromPoolAsync.
  //! @param __bytes The size in bytes of the allocation.
  //! @param __stream Stream on which to perform allocation.
  //! @throws cuda::cuda_error If an error code was return by the cuda api call.
  //! @returns Pointer to the newly allocated memory.
  [[nodiscard]] _CCCL_HOST_API void* allocate(const ::cuda::stream_ref __stream, const size_t __bytes)
  {
    ::CUdeviceptr __ptr = ::cuda::__driver::__mallocFromPoolAsync(__bytes, __pool_, __stream.get());
    return reinterpret_cast<void*>(__ptr); // NOLINT(performance-no-int-to-ptr)
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate`
  //! @param __bytes The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __alignment The alignment that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the
  //! stream used in the <a
  //! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a>
  //! call that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream
  //! other than \p __stream. It is the caller's responsibility to properly
  //! synchronize all relevant streams before calling `deallocate`.
  void _CCCL_HOST_API deallocate(
    const ::cuda::stream_ref __stream,
    void* __ptr,
    const size_t __bytes,
    [[maybe_unused]] const size_t __alignment) noexcept
  {
    // We need to ensure that the provided alignment matches the minimal
    // provided alignment
    _CCCL_ASSERT(__is_valid_alignment(__alignment), "Invalid alignment passed to __memory_pool_base::deallocate.");
    deallocate(__stream, __ptr, __bytes);
  }

  //! @brief Tries to release memory.
  //! @param __min_bytes_to_keep the minimal guaranteed size of the pool.
  //! @note If the pool has less than \p __minBytesToKeep reserved, the trim_to
  //! operation is a no-op. Otherwise the pool will be guaranteed to have at
  //! least \p __minBytesToKeep bytes reserved after the operation.
  _CCCL_HOST_API void trim_to(const size_t __min_bytes_to_keep)
  {
    ::cuda::__driver::__mempoolTrimTo(__pool_, __min_bytes_to_keep);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attr the attribute to get.
  //! @return The value of the attribute.
  template <typename _Attr>
  [[nodiscard]] _CCCL_HOST_API auto attribute(_Attr __attr) const
  {
    return __attr(__pool_);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attribute the attribute to get.
  //! @return The value of the attribute.
  template <::cudaMemPoolAttr _Attr>
  _CCCL_HOST_API auto attribute() const
  {
    return attribute(__pool_attr<_Attr>());
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <typename _Attr>
  _CCCL_HOST_API void set_attribute(_Attr __attr, typename _Attr::type __value)
  {
    __attr.set(__pool_, __value);
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <::cudaMemPoolAttr _Attr>
  _CCCL_HOST_API void set_attribute(typename __pool_attr<_Attr>::type __value)
  {
    return set_attribute(__pool_attr<_Attr>(), __value);
  }

  //! @brief Returns the underlying handle to the CUDA memory pool.
  [[nodiscard]] _CCCL_HOST_API constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_;
  }

  //! @brief Deallocate memory pointed to by \p __ptr.
  //! @param __ptr Pointer to be deallocated. Must have been allocated through a
  //! call to `allocate`.
  //! @param __bytes The number of bytes that was passed to the allocation call
  //! that returned \p __ptr.
  //! @param __stream A stream that has a stream ordering relationship with the
  //! stream used in the <a
  //! href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">allocate</a>
  //! call that returned \p __ptr.
  //! @note The pointer passed to `deallocate` must not be in use in a stream
  //! other than \p __stream. It is the caller's responsibility to properly
  //! synchronize all relevant streams before calling `deallocate`.
  _CCCL_HOST_API void deallocate(const ::cuda::stream_ref __stream, void* __ptr, size_t) noexcept
  {
    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeAsyncNoThrow, "deallocate failed", reinterpret_cast<::CUdeviceptr>(__ptr), __stream.get());
  }

  //! @brief Enable access to memory allocated through this memory resource by
  //! the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the
  //! span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to enable access
  //! for
  _CCCL_HOST_API void enable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::__mempool_set_access(__pool_, {__devices.data(), __devices.size()}, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }

  //! @brief Enable access to memory allocated through this memory resource by
  //! the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should
  //! be enabled
  _CCCL_HOST_API void enable_access_from(device_ref __device)
  {
    ::cuda::__mempool_set_access(__pool_, {&__device, 1}, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }

  //! @brief Disable access to memory allocated through this memory resource by
  //! the supplied devices
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //! Device on which this resource allocates memory can be included in the
  //! span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to disable access
  //! for
  _CCCL_HOST_API void disable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::__mempool_set_access(__pool_, {__devices.data(), __devices.size()}, ::CU_MEM_ACCESS_FLAGS_PROT_NONE);
  }

  //! @brief Disable access to memory allocated through this memory resource by
  //! the supplied device
  //!
  //! Access is controlled through the underlying memory pool, so this
  //! setting is shared between all memory resources created from the same pool.
  //!
  //! @param __device device_ref indicating for which device the access should
  //! be disabled
  _CCCL_HOST_API void disable_access_from(device_ref __device)
  {
    ::cuda::__mempool_set_access(__pool_, {&__device, 1}, ::CU_MEM_ACCESS_FLAGS_PROT_NONE);
  }

  //! @brief Query if memory allocated through this memory resource is
  //! accessible by the supplied device
  //!
  //! @param __device device for which the access is queried
  [[nodiscard]] _CCCL_HOST_API bool is_accessible_from(device_ref __device)
  {
    return ::cuda::__mempool_get_access(__pool_, __device);
  }

  //! @brief Equality comparison with another __memory_pool_base.
  //! @returns true if underlying \c cudaMemPool_t are equal.
  [[nodiscard]] _CCCL_HOST_API bool operator==(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_ == __rhs.__pool_;
  }

#  if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another __memory_pool_base.
  //! @returns true if underlying \c cudaMemPool_t are not equal.
  [[nodiscard]] _CCCL_HOST_API bool operator!=(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_ != __rhs.__pool_;
  }
#  endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA

_CCCL_DIAG_POP

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_RESOURCE_MEMORY_POOL_BASE_H
