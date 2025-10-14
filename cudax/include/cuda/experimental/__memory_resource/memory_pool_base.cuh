//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH
#define _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <stdexcept>

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/__device/attributes.h>
#include <cuda/__device/device_ref.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/span>

#include <cuda/experimental/__stream/internal_streams.cuh>

#include <cuda/std/__cccl/prologue.h>

//! @file
//! The \c __memory_pool_base class provides a wrapper around a `cudaMempool_t`.
namespace cuda::experimental
{

namespace __detail
{

enum class __pool_attr_settable : bool
{
};

template <::cudaMemPoolAttr _Attr, typename _Type, __pool_attr_settable _Settable>
struct __pool_attr_impl
{
  using type = _Type;

  [[nodiscard]] _CCCL_HOST_API constexpr operator ::cudaMemPoolAttr() const noexcept
  {
    return _Attr;
  }

  [[nodiscard]] _CCCL_HOST_API type operator()(::cudaMemPool_t __pool) const
  {
    size_t __value = ::cuda::__driver::__mempoolGetAttribute(__pool, static_cast<::CUmemPool_attribute>(_Attr));
    return static_cast<type>(__value);
  }

  static void set(::cudaMemPool_t __pool, type __value)
  {
    size_t __value_copy = __value;
    if constexpr (_Settable == __pool_attr_settable{true})
    {
      ::cuda::__driver::__mempoolSetAttribute(__pool, static_cast<::CUmemPool_attribute>(_Attr), &__value_copy);
    }
    else
    {
      ::cuda::std::__throw_invalid_argument("This attribute can't be set");
    }
  }
};

template <::cudaMemPoolAttr _Attr>
struct __pool_attr : __pool_attr_impl<_Attr, size_t, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::cudaMemPoolReuseFollowEventDependencies>
    : __pool_attr_impl<::cudaMemPoolReuseFollowEventDependencies, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::cudaMemPoolReuseAllowOpportunistic>
    : __pool_attr_impl<::cudaMemPoolReuseAllowOpportunistic, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::cudaMemPoolReuseAllowInternalDependencies>
    : __pool_attr_impl<::cudaMemPoolReuseAllowInternalDependencies, bool, __pool_attr_settable{true}>
{};

template <>
struct __pool_attr<::cudaMemPoolAttrReservedMemCurrent>
    : __pool_attr_impl<::cudaMemPoolAttrReservedMemCurrent, size_t, __pool_attr_settable{false}>
{};

template <>
struct __pool_attr<::cudaMemPoolAttrUsedMemCurrent>
    : __pool_attr_impl<::cudaMemPoolAttrUsedMemCurrent, size_t, __pool_attr_settable{false}>
{};

inline void __set_attribute_non_zero_only(::cudaMemPool_t __pool, ::CUmemPool_attribute __attr, size_t __value)
{
  if (__value != 0)
  {
    ::cuda::std::__throw_invalid_argument("This attribute can't be set to a non-zero value.");
  }
  ::cuda::__driver::__mempoolSetAttribute(__pool, __attr, &__value);
}

template <>
struct __pool_attr<::cudaMemPoolAttrReservedMemHigh>
    : __pool_attr_impl<::cudaMemPoolAttrReservedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::cudaMemPool_t __pool, type __value)
  {
    ::cuda::experimental::__detail::__set_attribute_non_zero_only(__pool, ::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, __value);
  }
};

template <>
struct __pool_attr<::cudaMemPoolAttrUsedMemHigh>
    : __pool_attr_impl<::cudaMemPoolAttrUsedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::cudaMemPool_t __pool, type __value)
  {
    ::cuda::experimental::__detail::__set_attribute_non_zero_only(__pool, ::CU_MEMPOOL_ATTR_USED_MEM_HIGH, __value);
  }
};

} // namespace __detail

namespace memory_pool_attributes
{
// The threshold at which the pool will release memory.
using release_threshold_t = __detail::__pool_attr<::cudaMemPoolAttrReleaseThreshold>;
static constexpr release_threshold_t release_threshold{};

// Allow the pool to reuse the memory across streams as long as there is a stream ordering dependency between the
// streams.
using reuse_follow_event_dependencies_t = __detail::__pool_attr<::cudaMemPoolReuseFollowEventDependencies>;
static constexpr reuse_follow_event_dependencies_t reuse_follow_event_dependencies{};

// Allow the pool to reuse already completed frees when there is no dependency between the streams.
using reuse_allow_opportunistic_t = __detail::__pool_attr<::cudaMemPoolReuseAllowOpportunistic>;
static constexpr reuse_allow_opportunistic_t reuse_allow_opportunistic{};

// Allow the pool to insert stream dependencies to reuse the memory across streams.
using reuse_allow_internal_dependencies_t = __detail::__pool_attr<::cudaMemPoolReuseAllowInternalDependencies>;
static constexpr reuse_allow_internal_dependencies_t reuse_allow_internal_dependencies{};

// The current amount of memory reserved in the pool.
using reserved_mem_current_t = __detail::__pool_attr<::cudaMemPoolAttrReservedMemCurrent>;
static constexpr reserved_mem_current_t reserved_mem_current{};

// The high water mark for the reserved memory in the pool.
using reserved_mem_high_t = __detail::__pool_attr<::cudaMemPoolAttrReservedMemHigh>;
static constexpr reserved_mem_high_t reserved_mem_high{};

// The current amount of memory used in the pool.
using used_mem_current_t = __detail::__pool_attr<::cudaMemPoolAttrUsedMemCurrent>;
static constexpr used_mem_current_t used_mem_current{};

// The high water mark for the used memory in the pool.
using used_mem_high_t = __detail::__pool_attr<::cudaMemPoolAttrUsedMemHigh>;
static constexpr used_mem_high_t used_mem_high{};
}; // namespace memory_pool_attributes

//! @brief  Checks whether the current device supports \c cudaMallocAsync.
//! @param __device The id of the device for which to query support.
//! @throws cuda_error if \c cudaDeviceGetAttribute failed.
//! @returns true if \c cudaDevAttrMemoryPoolsSupported is not zero.
inline void __verify_device_supports_stream_ordered_allocations(device_ref __device)
{
  if (!__device.attribute(::cuda::device_attributes::memory_pools_supported))
  {
    ::cuda::__throw_cuda_error(::cudaErrorNotSupported, "cudaMallocAsync is not supported on the given device");
  }
}

//! @brief Check whether the specified `cudaMemAllocationHandleType` is supported on the present
//! CUDA driver/runtime version.
//! @param __device The id of the device to check for support.
//! @param __handle_type An IPC export handle type to check for support.
//! @throws cuda_error if the specified `cudaMemAllocationHandleType` is not supported on the specified device.
inline void __verify_device_supports_export_handle_type(
  device_ref __device, ::cudaMemAllocationHandleType __handle_type, ::CUmemLocation __location)
{
  if (__handle_type == ::cudaMemAllocationHandleType::cudaMemHandleTypeNone)
  {
    return;
  }
  if (__location.type != ::CU_MEM_LOCATION_TYPE_DEVICE
#if _CCCL_CTK_AT_LEAST(12, 6)
      && __location.type != ::CU_MEM_LOCATION_TYPE_HOST_NUMA
#endif
  )
  {
    ::cuda::__throw_cuda_error(
      ::cudaErrorNotSupported, "Requested IPC memory handle type not supported for the given location");
  }
  auto __supported_handles = __device.attribute(::cuda::device_attributes::memory_pool_supported_handle_types);
  if ((static_cast<int>(__handle_type) & __supported_handles) != static_cast<int>(__handle_type))
  {
    ::cuda::__throw_cuda_error(
      ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on a given device");
  }
}

//! @brief Enable access to this memory pool from the supplied devices
//!
//! Device on which this pool resides can be included in the span.
//!
//! @param __pool The memory pool to set access for
//! @param __devices A span of `device_ref`s listing devices to enable access for
//! @param __flags The access flags to set
//! @throws cuda_error if ``cudaMemPoolSetAccess`` fails.
inline void
__mempool_set_access(::CUmemoryPool __pool, ::cuda::std::span<const device_ref> __devices, ::CUmemAccess_flags __flags)
{
  ::std::vector<::CUmemAccessDesc> __descs;
  __descs.reserve(__devices.size());
  for (size_t __i = 0; __i < __devices.size(); ++__i)
  {
    __descs.push_back({::CUmemLocation{::CU_MEM_LOCATION_TYPE_DEVICE, __devices[__i].get()}, __flags});
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

//! @brief \c memory_pool_properties is a wrapper around properties passed to \c __memory_pool_base to create a
//! <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html">cudaMemPool_t</a>.
struct memory_pool_properties
{
  size_t initial_pool_size                           = 0;
  size_t release_threshold                           = ::cuda::std::numeric_limits<size_t>::max();
  cudaMemAllocationHandleType allocation_handle_type = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
};

class __memory_pool_base
{
private:
  ::cudaMemPool_t __pool_handle_ = nullptr;

  //! @brief  Creates the CUDA memory pool from the passed in arguments.
  //! @throws cuda_error If the creation of the CUDA memory pool failed.
  //! @returns The created CUDA memory pool.
  [[nodiscard]] static cudaMemPool_t __create_cuda_mempool(
    memory_pool_properties __properties, ::CUmemLocation __location, CUmemAllocationType __allocation_type) noexcept
  {
    ::CUmemPoolProps __pool_properties{};
    __pool_properties.allocType   = __allocation_type;
    __pool_properties.handleTypes = ::CUmemAllocationHandleType(__properties.allocation_handle_type);
    __pool_properties.location    = __location;

    if (__properties.initial_pool_size > __properties.release_threshold)
    {
      ::cuda::std::__throw_invalid_argument("Initial pool size must be less than the release threshold");
    }

    ::CUmemoryPool __cuda_pool_handle{};
    ::cudaError_t __error = ::cuda::__driver::__mempoolCreateNoThrow(&__cuda_pool_handle, &__pool_properties);
    if (__error != ::cudaSuccess)
    {
      // Mempool creation failed, lets try to figure out why
      ::cuda::experimental::__verify_device_supports_stream_ordered_allocations(__location.id);
      ::cuda::experimental::__verify_device_supports_export_handle_type(
        __location.id, __properties.allocation_handle_type, __location);

      // Could not find the reason, throw a generic error
      ::cuda::__throw_cuda_error(__error, "Failed to create a memory pool");
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
        ::cuda::__throw_cuda_error(::cudaErrorMemoryAllocation, "Failed to allocate initial pool size");
      }
    }
    return __cuda_pool_handle;
  }

protected:
  struct __from_handle_t
  {};

  //! @brief Constructs a \c __memory_pool_base from a handle taking ownership of the pool
  //! @param __handle The handle to the existing pool
  explicit __memory_pool_base(__from_handle_t, ::cudaMemPool_t __handle) noexcept
      : __pool_handle_(__handle)
  {}

public:
  //! @brief Constructs a \c __memory_pool_base with the optionally specified initial pool size and release threshold.
  //! If the pool size grows beyond the release threshold, unused memory held by the pool will be released at the next
  //! synchronization event.
  //! @throws cuda_error if the CUDA version does not support ``cudaMallocAsync``.
  //! @param __id The device id of the device the stream pool is constructed on.
  //! @param __pool_properties Optional, additional properties of the pool to be created.
  explicit __memory_pool_base(
    memory_pool_properties __properties, ::CUmemLocation __location, CUmemAllocationType __allocation_type)
      : __pool_handle_(__create_cuda_mempool(__properties, __location, __allocation_type))
  {}

  __memory_pool_base(__memory_pool_base const&)            = delete;
  __memory_pool_base(__memory_pool_base&&)                 = delete;
  __memory_pool_base& operator=(__memory_pool_base const&) = delete;
  __memory_pool_base& operator=(__memory_pool_base&&)      = delete;

  ~__memory_pool_base() noexcept
  {
    ::cuda::__driver::__mempoolDestroy(__pool_handle_);
  }

  //! @brief Tries to release memory.
  //! @param __min_bytes_to_keep the minimal guaranteed size of the pool.
  //! @note If the pool has less than \p __minBytesToKeep reserved, the trim_to operation is a no-op. Otherwise the
  //! pool will be guaranteed to have at least \p __minBytesToKeep bytes reserved after the operation.
  _CCCL_HOST_API void trim_to(const size_t __min_bytes_to_keep)
  {
    ::cuda::__driver::__mempoolTrimTo(__pool_handle_, __min_bytes_to_keep);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attr the attribute to get.
  //! @return The value of the attribute.
  template <typename _Attr>
  [[nodiscard]] _CCCL_HOST_API auto attribute(_Attr __attr) const
  {
    return __attr(__pool_handle_);
  }

  //! @brief Gets the value of an attribute of the pool.
  //! @param __attribute the attribute to get.
  //! @return The value of the attribute.
  template <::cudaMemPoolAttr _Attr>
  _CCCL_HOST_API auto attribute() const
  {
    return attribute(__detail::__pool_attr<_Attr>());
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <typename _Attr>
  _CCCL_HOST_API void set_attribute(_Attr __attr, typename _Attr::type __value)
  {
    __attr.set(__pool_handle_, __value);
  }

  //! @brief Sets an attribute of the pool to a given value.
  //! @param __attribute the attribute to be set.
  //! @param __value the new value of that attribute.
  template <::cudaMemPoolAttr _Attr>
  _CCCL_HOST_API void set_attribute(typename __detail::__pool_attr<_Attr>::type __value)
  {
    return set_attribute(__detail::__pool_attr<_Attr>(), __value);
  }

  //! @brief Enable access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to enable access for
  _CCCL_HOST_API void enable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_handle_, {__devices.data(), __devices.size()}, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }

  //! @brief Enable access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be enabled
  _CCCL_HOST_API void enable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_handle_, {&__device, 1}, ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }

  //! @brief Disable access to this memory pool from the supplied devices
  //!
  //! Device on which this pool resides can be included in the span.
  //!
  //! @param __devices A span of `device_ref`s listing devices to disable access for
  _CCCL_HOST_API void disable_access_from(::cuda::std::span<const device_ref> __devices)
  {
    ::cuda::experimental::__mempool_set_access(
      __pool_handle_, {__devices.data(), __devices.size()}, ::CU_MEM_ACCESS_FLAGS_PROT_NONE);
  }

  //! @brief Disable access to this memory pool from the supplied device
  //!
  //! @param __device device_ref indicating for which device the access should be disable
  _CCCL_HOST_API void disable_access_from(device_ref __device)
  {
    ::cuda::experimental::__mempool_set_access(__pool_handle_, {&__device, 1}, ::CU_MEM_ACCESS_FLAGS_PROT_NONE);
  }

  //! @brief Query if memory allocated through this memory resource is accessible by the supplied device
  //!
  //! @param __device device for which the access is queried
  [[nodiscard]] _CCCL_HOST_API bool is_accessible_from(device_ref __device)
  {
    return ::cuda::experimental::__mempool_get_access(__pool_handle_, __device);
  }

  //! @brief Equality comparison with another \c __memory_pool_base.
  //! @returns true if the stored ``cudaMemPool_t`` are equal.
  [[nodiscard]] _CCCL_HOST_API constexpr bool operator==(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_handle_ == __rhs.__pool_handle_;
  }

#if _CCCL_STD_VER <= 2017
  //! @brief Inequality comparison with another \c __memory_pool_base.
  //! @returns true if the stored ``cudaMemPool_t`` are not equal.
  [[nodiscard]] _CCCL_HOST_API constexpr bool operator!=(__memory_pool_base const& __rhs) const noexcept
  {
    return __pool_handle_ != __rhs.__pool_handle_;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Equality comparison with a \c cudaMemPool_t.
  //! @param __rhs A \c cudaMemPool_t.
  //! @returns true if the stored ``cudaMemPool_t`` is equal to \p __rhs.
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(__memory_pool_base const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ == __rhs;
  }

#if _CCCL_STD_VER <= 2017
  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(::cudaMemPool_t __lhs, __memory_pool_base const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ == __lhs;
  }

  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(__memory_pool_base const& __lhs, ::cudaMemPool_t __rhs) noexcept
  {
    return __lhs.__pool_handle_ != __rhs;
  }

  //! @copydoc __memory_pool_base::operator==(__memory_pool_base const&, ::cudaMemPool_t)
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(::cudaMemPool_t __lhs, __memory_pool_base const& __rhs) noexcept
  {
    return __rhs.__pool_handle_ != __lhs;
  }
#endif // _CCCL_STD_VER <= 2017

  //! @brief Returns the underlying handle to the CUDA memory pool.
  [[nodiscard]] _CCCL_HOST_API constexpr cudaMemPool_t get() const noexcept
  {
    return __pool_handle_;
  }
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_MEMORY_POOL_BASE_CUH
