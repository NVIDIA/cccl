//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_POOL_ATTRIBUTES_H
#define _CUDA___MEMORY_POOL_ATTRIBUTES_H

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
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/cstddef>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
// clang complains about missing braces in CUmemLocation constructor but GCC complains if we add them

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
      _CCCL_THROW(::std::invalid_argument, "This attribute can't be set");
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
    _CCCL_THROW(::std::invalid_argument, "This attribute can't be set to a non-zero value.");
  }
  ::cuda::__driver::__mempoolSetAttribute(__pool, __attr, &__value);
}

template <>
struct __pool_attr<::cudaMemPoolAttrReservedMemHigh>
    : __pool_attr_impl<::cudaMemPoolAttrReservedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::cudaMemPool_t __pool, type __value)
  {
    ::cuda::__set_attribute_non_zero_only(__pool, ::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, __value);
  }
};

template <>
struct __pool_attr<::cudaMemPoolAttrUsedMemHigh>
    : __pool_attr_impl<::cudaMemPoolAttrUsedMemHigh, size_t, __pool_attr_settable{true}>
{
  static void set(::cudaMemPool_t __pool, type __value)
  {
    ::cuda::__set_attribute_non_zero_only(__pool, ::CU_MEMPOOL_ATTR_USED_MEM_HIGH, __value);
  }
};

namespace memory_pool_attributes
{
// The threshold at which the pool will release memory.
using release_threshold_t = __pool_attr<::cudaMemPoolAttrReleaseThreshold>;
static constexpr release_threshold_t release_threshold{};

// Allow the pool to reuse the memory across streams as long as there is a
// stream ordering dependency between the streams.
using reuse_follow_event_dependencies_t = __pool_attr<::cudaMemPoolReuseFollowEventDependencies>;
static constexpr reuse_follow_event_dependencies_t reuse_follow_event_dependencies{};

// Allow the pool to reuse already completed frees when there is no dependency
// between the streams.
using reuse_allow_opportunistic_t = __pool_attr<::cudaMemPoolReuseAllowOpportunistic>;
static constexpr reuse_allow_opportunistic_t reuse_allow_opportunistic{};

// Allow the pool to insert stream dependencies to reuse the memory across
// streams.
using reuse_allow_internal_dependencies_t = __pool_attr<::cudaMemPoolReuseAllowInternalDependencies>;
static constexpr reuse_allow_internal_dependencies_t reuse_allow_internal_dependencies{};

// The current amount of memory reserved in the pool.
using reserved_mem_current_t = __pool_attr<::cudaMemPoolAttrReservedMemCurrent>;
static constexpr reserved_mem_current_t reserved_mem_current{};

// The high water mark for the reserved memory in the pool.
using reserved_mem_high_t = __pool_attr<::cudaMemPoolAttrReservedMemHigh>;
static constexpr reserved_mem_high_t reserved_mem_high{};

// The current amount of memory used in the pool.
using used_mem_current_t = __pool_attr<::cudaMemPoolAttrUsedMemCurrent>;
static constexpr used_mem_current_t used_mem_current{};

// The high water mark for the used memory in the pool.
using used_mem_high_t = __pool_attr<::cudaMemPoolAttrUsedMemHigh>;
static constexpr used_mem_high_t used_mem_high{};
}; // namespace memory_pool_attributes

inline bool __is_host_memory_pool_supported()
{
  // Both host_numa and host memory pool flags should agree, but check the one corresponding to the implementation
  // of the default pool just to be sure
#  if _CCCL_CTK_AT_LEAST(13, 0)
  return ::cuda::device_attributes::host_memory_pools_supported(cuda::device_ref{0});
#  elif _CCCL_CTK_AT_LEAST(12, 9)
  return ::cuda::device_attributes::host_numa_memory_pools_supported(cuda::device_ref{0});
#  else
  return false;
#  endif
}

//! @brief  Checks whether the current device supports stream-ordered
//! allocations.
//! @param __device The device for which to query support.
//! @throws cuda_error if \c cudaDeviceGetAttribute failed.
//! @returns true if \c cudaDevAttrMemoryPoolsSupported is not zero.
inline void __verify_device_supports_stream_ordered_allocations(
  ::CUmemLocation __location, [[maybe_unused]] ::CUmemAllocationType __allocation_type)
{
  auto __device =
    __location.type == ::CU_MEM_LOCATION_TYPE_DEVICE ? cuda::device_ref{__location.id} : cuda::device_ref{0};
  if (!::cuda::device_attributes::memory_pools_supported(__device))
  {
    _CCCL_THROW(::cuda::cuda_error, ::cudaErrorNotSupported, "stream-ordered allocations are not supported");
  }
#  if _CCCL_CTK_AT_LEAST(13, 0)
  if (__allocation_type == ::CU_MEM_ALLOCATION_TYPE_MANAGED
      && !::cuda::device_attributes::concurrent_managed_access(__device))
  {
    _CCCL_THROW(::cuda::cuda_error, ::cudaErrorNotSupported, "managed memory pools are not supported");
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#  if _CCCL_CTK_AT_LEAST(12, 9)
  if (__location.type == ::CU_MEM_LOCATION_TYPE_HOST && !__is_host_memory_pool_supported())
  {
    _CCCL_THROW(::cuda::cuda_error, ::cudaErrorNotSupported, "host memory pools are not supported");
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 9)
}

//! @brief Check whether the specified `cudaMemAllocationHandleType` is
//! supported on the present CUDA driver/runtime version.
//! @param __device The device to check for support.
//! @param __handle_type An IPC export handle type to check for support.
//! @throws cuda_error if the specified `cudaMemAllocationHandleType` is not
//! supported on the specified device.
inline void __verify_device_supports_export_handle_type(
  const device_ref __device, ::cudaMemAllocationHandleType __handle_type, ::CUmemLocation __location)
{
  if (__handle_type == ::cudaMemAllocationHandleType::cudaMemHandleTypeNone)
  {
    return;
  }
  if (__location.type != ::CU_MEM_LOCATION_TYPE_DEVICE
#  if _CCCL_CTK_AT_LEAST(12, 9)
      && __location.type != ::CU_MEM_LOCATION_TYPE_HOST_NUMA
#  endif
  )
  {
    _CCCL_THROW(::cuda::cuda_error,
                ::cudaErrorNotSupported,
                "Requested IPC memory handle type not supported "
                "for the given location");
  }
  auto __supported_handles = __device.attribute(::cuda::device_attributes::memory_pool_supported_handle_types);
  if ((static_cast<int>(__handle_type) & __supported_handles) != static_cast<int>(__handle_type))
  {
    _CCCL_THROW(
      cuda::cuda_error, ::cudaErrorNotSupported, "Requested IPC memory handle type not supported on a given device");
  }
}

[[nodiscard]] _CCCL_HOST_API inline cudaMemPool_t
__get_default_memory_pool(const CUmemLocation __location, [[maybe_unused]] const CUmemAllocationType __allocation_type)
{
  ::cuda::__verify_device_supports_stream_ordered_allocations(__location, __allocation_type);

#  if _CCCL_CTK_AT_LEAST(13, 0)
  ::cudaMemPool_t __pool = ::cuda::__driver::__getDefaultMemPool(__location, __allocation_type);
  if (::cuda::memory_pool_attributes::release_threshold(__pool) == 0)
  {
    ::cuda::memory_pool_attributes::release_threshold.set(__pool, ::cuda::std::numeric_limits<size_t>::max());
  }
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  _CCCL_ASSERT(__location.type == ::CU_MEM_LOCATION_TYPE_DEVICE,
               "Before CUDA 13 only device memory pools have a default");
  ::cudaMemPool_t __pool = ::cuda::__driver::__deviceGetDefaultMemPool(::CUdevice{__location.id});
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^
  return __pool;
}

_CCCL_END_NAMESPACE_CUDA

_CCCL_DIAG_POP

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___MEMORY_POOL_ATTRIBUTES_H
