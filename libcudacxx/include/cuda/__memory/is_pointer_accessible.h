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

#include <cuda/__driver/driver_api.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_pointer.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

_CCCL_TEMPLATE(typename _Pointer)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Pointer> || ::cuda::std::is_pointer_v<_Pointer>)
[[nodiscard]]
_CCCL_HOST_API bool is_managed_pointer(_Pointer __p)
{
  const auto __p1 = ::cuda::std::to_address(__p);
  bool __is_managed{};
  const auto __status =
    ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_IS_MANAGED>(__is_managed, __p1);
  if (__status != ::cudaErrorInvalidValue && __status != ::cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "is_managed_pointer failed()", _CCCL_BUILTIN_PRETTY_FUNCTION());
  }
  return (__status == ::cudaErrorInvalidValue) || __is_managed;
}

_CCCL_TEMPLATE(typename _Pointer)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Pointer> || ::cuda::std::is_pointer_v<_Pointer>)
[[nodiscard]]
_CCCL_HOST_API bool is_host_accessible(_Pointer __p)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    const auto __p1 = ::cuda::std::to_address(__p);
    ::CUmemorytype __type{};
    const auto __status =
      ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(__type, __p1);
    if (__status != ::cudaErrorInvalidValue && __status != ::cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__status, "is_host_accessible failed()", _CCCL_BUILTIN_PRETTY_FUNCTION());
    }
    return (__status == ::cudaErrorInvalidValue)
        || (__type == ::CU_MEMORYTYPE_HOST || __type == ::CU_MEMORYTYPE_UNIFIED)
        || ::cuda::is_managed_pointer(__p); // needed because MEMORY_TYPE is not sufficient for managed memory
  }
  return true; // cannot be verified
}

_CCCL_TEMPLATE(typename _Pointer)
_CCCL_REQUIRES(::cuda::std::contiguous_iterator<_Pointer> || ::cuda::std::is_pointer_v<_Pointer>)
[[nodiscard]]
_CCCL_HOST_API bool is_device_accessible(_Pointer __p)
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    const auto __p1 = ::cuda::std::to_address(__p);
    ::CUmemorytype __type{};
    const auto __status =
      ::cuda::__driver::__pointerGetAttributeNoThrow<::CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(__type, __p1);
    if (__status != ::cudaErrorInvalidValue && __status != ::cudaSuccess)
    {
      ::cuda::__throw_cuda_error(__status, "is_device_accessible failed()", _CCCL_BUILTIN_PRETTY_FUNCTION());
    }
    return (__status == ::cudaErrorInvalidValue)
        || (__type == ::CU_MEMORYTYPE_DEVICE || __type == ::CU_MEMORYTYPE_UNIFIED)
        || ::cuda::is_managed_pointer(__p); // needed because ATTRIBUTE_MEMORY_TYPE is not sufficient for managed memory
  }
  return true; // cannot be verified
}

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_POINTER_ACCESSIBLE_H
