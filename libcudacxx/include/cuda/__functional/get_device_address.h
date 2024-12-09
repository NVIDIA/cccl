//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___GET_DEVICE_ADDRESS_H
#define _CUDA___GET_DEVICE_ADDRESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__memory/addressof.h>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Returns the device address of the passed \c __device_object
//! @param __device_object the object residing in device memory
//! @return Valid pointer to the device object
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _Tp* get_device_address(_Tp& __device_object)
{
#if _CCCL_HAS_CUDA_COMPILER
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (return _CUDA_VSTD::addressof(__device_object);),
    (void* __device_ptr = nullptr; _CCCL_TRY_CUDA_API(
       ::cudaGetSymbolAddress,
       "failed to call cudaGetSymbolAddress in cuda::get_device_address",
       &__device_ptr,
       __device_object);
     return static_cast<_Tp*>(__device_ptr);))
#else // ^^^ _CCCL_HAS_CUDA_COMPILER ^^^ / vvv !_CCCL_HAS_CUDA_COMPILER vvv
  return _CUDA_VSTD::addressof(__device_object);
#endif // !_CCCL_HAS_CUDA_COMPILER
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___GET_DEVICE_ADDRESS_H
