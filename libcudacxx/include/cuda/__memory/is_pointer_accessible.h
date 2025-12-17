//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_IS_POINTER_ACCESSIBLE_H
#define _CUDA___MEMORY_IS_POINTER_ACCESSIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__type_traits/integral_constant.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  define _CCCL_THROW_OR_RETURN(_STATUS, _MSG)                                          \
    if ((_STATUS) != ::cudaSuccess)                                                     \
    {                                                                                   \
      if constexpr (_IsNothrow)                                                         \
      {                                                                                 \
        return false;                                                                   \
      }                                                                                 \
      else                                                                              \
      {                                                                                 \
        ::cuda::__throw_cuda_error((_STATUS), (_MSG), _CCCL_BUILTIN_PRETTY_FUNCTION()); \
      }                                                                                 \
    }

template <bool _IsNothrow>
[[nodiscard]]
_CCCL_HOST_API inline bool __is_managed(const void* __p, ::cuda::std::bool_constant<_IsNothrow>) noexcept(_IsNothrow)
{
  if (__p == nullptr)
  {
    return false;
  }
  bool __is_managed{};
  const auto __status =
    ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_IS_MANAGED>(__is_managed, __p);
  switch (__status)
  {
    case ::cudaSuccess:
      return __is_managed;
    case ::cudaErrorInvalidValue:
      return false;
    default:
      _CCCL_THROW_OR_RETURN(__status, "is_managed() failed");
      return false;
  }
}

/**
 * @brief Checks if a pointer is a managed pointer.
 *
 * @param __p The pointer to check.
 * @return `true` if the pointer is a managed pointer, `false` otherwise.
 */
[[nodiscard]]
_CCCL_HOST_API inline bool is_managed(const void* __p)
{
  return ::cuda::__is_managed(__p, ::cuda::std::false_type{});
}

[[nodiscard]]
_CCCL_HOST_API inline bool __is_managed_nothrow(const void* __p) noexcept
{
  return ::cuda::__is_managed(__p, ::cuda::std::true_type{});
}

template <bool _IsNothrow>
[[nodiscard]]
_CCCL_HOST_API inline bool
__is_host_accessible(const void* __p, ::cuda::std::bool_constant<_IsNothrow>) noexcept(_IsNothrow)
{
  if (__p == nullptr)
  {
    return false;
  }
  ::CUpointer_attribute __attrs[3] = {
    ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ::CU_POINTER_ATTRIBUTE_IS_MANAGED, ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE};
  auto __memory_type       = static_cast<::CUmemorytype>(0);
  int __is_managed         = 0;
  ::CUmemoryPool __mempool = nullptr;
  void* __results[3]       = {&__memory_type, &__is_managed, &__mempool};
  const auto __status      = ::cuda::__driver::__pointerGetAttributesNoThrow(__attrs, __results, __p);
  _CCCL_THROW_OR_RETURN(__status, "Failed to get attributes of a pointer");
  // (1) check if the pointer is unregistered
  if (__memory_type == static_cast<::CUmemorytype>(0)
      || (__mempool == nullptr && (__is_managed || __memory_type == ::CU_MEMORYTYPE_HOST)))
  {
    return true;
  }
  // (2) check if a memory pool is associated with the pointer
#  if _CCCL_CTK_AT_LEAST(12, 2)
  if (__mempool != nullptr)
  {
    ::CUmemLocation __prop{::CU_MEM_LOCATION_TYPE_HOST, 0};
    ::CUmemAccess_flags __pool_flags;
    const auto __status2 = ::cuda::__driver::__mempoolGetAccessNoThrow(__pool_flags, __mempool, &__prop);
    _CCCL_THROW_OR_RETURN(__status2, "Failed to get access of a memory pool");
    return __pool_flags & unsigned{::CU_MEM_ACCESS_FLAGS_PROT_READ};
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 2)
  return false;
}

/**
 * @brief Checks if a pointer is a host accessible pointer.
 *
 * @param __p The pointer to check.
 * @return `true` if the pointer is a host accessible pointer, `false` otherwise.
 */
[[nodiscard]]
_CCCL_HOST_API inline bool is_host_accessible(const void* __p)
{
  return ::cuda::__is_host_accessible(__p, ::cuda::std::false_type{});
}

[[nodiscard]]
_CCCL_HOST_API inline bool __is_host_accessible_nothrow(const void* __p) noexcept
{
  return ::cuda::__is_host_accessible(__p, ::cuda::std::true_type{});
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
// clang complains about missing braces in CUmemLocation constructor but GCC complains if we add them

/**
 * @brief Checks if a pointer is a device pointer.
 *
 * This internal-only function can be used when the device id is not known.
 * The main difference between this function and is_device_accessible() is that this function does not check if the
 * pointer is peer accessible from a specified device.
 *
 * @param __p The pointer to check.
 * @return `true` if the pointer is a device pointer, `false` otherwise.
 */
[[nodiscard]]
_CCCL_HOST_API inline bool __is_device_or_managed_memory(const void* __p) noexcept
{
  if (__p == nullptr)
  {
    return false;
  }
  ::CUpointer_attribute __attrs[4] = {
    ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    ::CU_POINTER_ATTRIBUTE_IS_MANAGED,
    ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE};
  auto __memory_type       = static_cast<::CUmemorytype>(0);
  int __is_managed         = 0;
  int __ptr_dev_id         = 0;
  ::CUmemoryPool __mempool = nullptr;
  void* __results[4]       = {&__memory_type, &__is_managed, &__ptr_dev_id, &__mempool};
  const auto __status      = ::cuda::__driver::__pointerGetAttributesNoThrow(__attrs, __results, __p);
  if (__status != ::cudaSuccess)
  {
    return false;
  }
  // (1) check if the pointer is unregistered
  if (__memory_type == static_cast<::CUmemorytype>(0))
  {
    return false;
  }
  // (2) check if a memory pool is associated with the pointer
  if (__mempool != nullptr)
  {
    ::CUmemLocation __prop{::CU_MEM_LOCATION_TYPE_DEVICE, __ptr_dev_id};
    ::CUmemAccess_flags __pool_flags;
    const auto __status2 = ::cuda::__driver::__mempoolGetAccessNoThrow(__pool_flags, __mempool, &__prop);
    return (__status2 == ::cudaSuccess) && (static_cast<bool>(__pool_flags));
  }
  // (3) check if the pointer is a device accessible pointer or managed memory
  return __is_managed || __memory_type == ::CU_MEMORYTYPE_DEVICE;
}

template <bool _IsNothrow>
[[nodiscard]]
_CCCL_HOST_API inline bool __is_device_accessible(
  const void* __p, device_ref __device, ::cuda::std::bool_constant<_IsNothrow>) noexcept(_IsNothrow)
{
  if (__p == nullptr)
  {
    return false;
  }
  ::CUpointer_attribute __attrs[4] = {
    ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    ::CU_POINTER_ATTRIBUTE_IS_MANAGED,
    ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE};
  auto __memory_type       = static_cast<::CUmemorytype>(0);
  int __is_managed         = 0;
  int __ptr_dev_id         = 0;
  ::CUmemoryPool __mempool = nullptr;
  void* __results[4]       = {&__memory_type, &__is_managed, &__ptr_dev_id, &__mempool};
  const auto __status      = ::cuda::__driver::__pointerGetAttributesNoThrow(__attrs, __results, __p);
  _CCCL_THROW_OR_RETURN(__status, "Failed to get attributes of a pointer");
  // (1) check if the pointer is unregistered
  if (__memory_type == static_cast<::CUmemorytype>(0))
  {
    return false;
  }
  // (2) check if the pointer is a device accessible pointer or managed memory
  if (!__is_managed && __memory_type != ::CU_MEMORYTYPE_DEVICE)
  {
    return false;
  }
  // (3) check if a memory pool is associated with the pointer
  if (__mempool != nullptr)
  {
    ::CUmemLocation __prop{::CU_MEM_LOCATION_TYPE_DEVICE, __device.get()};
    ::CUmemAccess_flags __pool_flags;
    const auto __status2 = ::cuda::__driver::__mempoolGetAccessNoThrow(__pool_flags, __mempool, &__prop);
    _CCCL_THROW_OR_RETURN(__status2, "Failed to get access of a memory pool");
    return __pool_flags & unsigned{::CU_MEM_ACCESS_FLAGS_PROT_READ};
  }
  // (4) check if the pointer is allocated on the specified device
  if (__ptr_dev_id == __device.get())
  {
    return true;
  }
  // (5) check if the pointer is peer accessible from the specified device
  int __result         = 0;
  const auto __status3 = ::cuda::__driver::__deviceCanAccessPeerNoThrow(__result, __device.get(), __ptr_dev_id);
  _CCCL_THROW_OR_RETURN(__status3, "Failed to check if the pointer is peer accessible from the specified device");
  return static_cast<bool>(__result);
}

/**
 * @brief Checks if a pointer is a device accessible pointer.
 *
 * @param __p The pointer to check.
 * @param __device The device to check.
 * @return `true` if the pointer is a device accessible pointer, `false` otherwise.
 */
[[nodiscard]]
_CCCL_HOST_API inline bool is_device_accessible(const void* __p, device_ref __device)
{
  return ::cuda::__is_device_accessible(__p, __device, ::cuda::std::false_type{});
}

[[nodiscard]]
_CCCL_HOST_API inline bool __is_device_accessible_nothrow(const void* __p, device_ref __device) noexcept
{
  return ::cuda::__is_device_accessible(__p, __device, ::cuda::std::true_type{});
}

_CCCL_DIAG_POP

#  undef _CCCL_THROW_OR_RETURN

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_POINTER_ACCESSIBLE_H
