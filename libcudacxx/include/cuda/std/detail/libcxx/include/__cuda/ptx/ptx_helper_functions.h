// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_HELPER_FUNCTIONS_H_
#define _CUDA_PTX_HELPER_FUNCTIONS_H_

#include "../../cstdint"        // uint32_t

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

extern "C" _LIBCUDACXX_DEVICE void __cuda_ptx__as_conv_unavailable_in_host_code();

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_ptr_smem(const void* __ptr)
{
  // Consider adding debug asserts here.
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      return static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__ptr));
    ),
    NV_ANY_TARGET, (
      __ptr;
      __cuda_ptx__as_conv_unavailable_in_host_code();
    )
  )
}

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_ptr_remote_dsmem(const void* __ptr)
{
  // No difference in implementation to __as_ptr_smem.
  // Consider adding debug asserts here.
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      return static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__ptr));
    ),
    NV_ANY_TARGET, (
      __ptr;
      __cuda_ptx__as_conv_unavailable_in_host_code();
    )
  )
}

inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_ptr_gmem(const void* __ptr)
{
  // Consider adding debug asserts here.
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      return static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__ptr));
    ),
    NV_ANY_TARGET, (
      __ptr;
      __cuda_ptx__as_conv_unavailable_in_host_code();
    )
  )
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint32_t __as_b32(_Tp __val)
{
  static_assert(sizeof(_Tp) == 4, "");
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      // Consider using std::bitcast
      return *reinterpret_cast<_CUDA_VSTD::uint32_t*>(&__val);
    ),
    NV_ANY_TARGET, (
      __cuda_ptx__as_conv_unavailable_in_host_code();
    )
  )
}

template <typename _Tp>
inline _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t __as_b64(_Tp __val)
{
  static_assert(sizeof(_Tp) == 8, "");
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (
      // Consider using std::bitcast
      return *reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__val);
    ),
    NV_ANY_TARGET, (
      __cuda_ptx__as_conv_unavailable_in_host_code();
    )
  )
}

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_HELPER_FUNCTIONS_H_
