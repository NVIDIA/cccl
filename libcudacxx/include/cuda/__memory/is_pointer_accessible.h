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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

[[nodiscard]]
_CCCL_HOST_API inline bool is_managed(const void* __p)
{
  bool __is_managed{};
  const auto __status =
    ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_IS_MANAGED>(__is_managed, __p);
  if (__status != ::cudaErrorInvalidValue && __status != ::cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "is_managed() failed", _CCCL_BUILTIN_PRETTY_FUNCTION());
  }
  return (__status == ::cudaErrorInvalidValue) || __is_managed;
}

[[nodiscard]]
_CCCL_HOST_API inline bool is_host_accessible(const void* __p)
{
  ::CUpointer_attribute __attrs[2] = {::CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ::CU_POINTER_ATTRIBUTE_IS_MANAGED};
  ::CUmemorytype __memory_type     = static_cast<::CUmemorytype>(0);
  int __is_managed                 = 0;
  void* __results[2]               = {&__memory_type, &__is_managed};
  const auto __status              = ::cuda::__driver::__pointerGetAttributesNoThrow(
    ::cuda::std::span<::CUpointer_attribute, 2>{__attrs}, ::cuda::std::span<void*, 2>{__results}, __p);
  if (__status != ::cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "is_host_accessible() failed", _CCCL_BUILTIN_PRETTY_FUNCTION());
  }
  if (__memory_type == static_cast<::CUmemorytype>(0)) // check if the pointer is unregistered
  {
    return true;
  }
  return (__is_managed || __memory_type == ::CU_MEMORYTYPE_UNIFIED || __memory_type == ::CU_MEMORYTYPE_HOST);
}

[[nodiscard]]
_CCCL_HOST_API inline bool is_device_accessible(const void* __p, device_ref __device)
{
  ::CUpointer_attribute __attrs[4] = {
    ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
    ::CU_POINTER_ATTRIBUTE_IS_MANAGED,
    ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
    ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE};
  ::CUmemorytype __memory_type = static_cast<::CUmemorytype>(0);
  int __is_managed             = 0;
  int __ptr_dev_id             = 0;
  ::CUmemoryPool __ptr_mempool = nullptr;
  void* __results[4]           = {&__memory_type, &__is_managed, &__ptr_dev_id, &__ptr_mempool};
  const auto __status          = ::cuda::__driver::__pointerGetAttributesNoThrow(
    ::cuda::std::span<::CUpointer_attribute, 4>{__attrs}, ::cuda::std::span<void*, 4>{__results}, __p);
  if (__status != ::cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "is_device_accessible() failed", _CCCL_BUILTIN_PRETTY_FUNCTION());
  }
  // (1) check if the pointer is unregistered
  if (__memory_type == static_cast<::CUmemorytype>(0))
  {
    return true;
  }
  // (2) check if the pointer is a device accessible pointer or managed memory
  if (!__is_managed && __memory_type != ::CU_MEMORYTYPE_DEVICE && __memory_type != ::CU_MEMORYTYPE_UNIFIED)
  {
    return false;
  }
  // (3) check if a memory pool is associated with the pointer
  if (__ptr_mempool != nullptr)
  {
    ::CUmemLocation __prop{::CU_MEM_LOCATION_TYPE_DEVICE, __device.get()};
    const auto __pool_flags = ::cuda::__driver::__mempoolGetAccess(__ptr_mempool, &__prop);
    return (__pool_flags == ::CU_MEM_ACCESS_FLAGS_PROT_READ || __pool_flags == ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
  }
  // (4) check if the pointer is allocated on the specified device
  if (__ptr_dev_id == __device.get())
  {
    return true;
  }
  // (5) check if the pointer is peer accessible from the specified device
  return ::cuda::__driver::__deviceCanAccessPeer(__device.get(), __ptr_dev_id);
}

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_POINTER_ACCESSIBLE_H
