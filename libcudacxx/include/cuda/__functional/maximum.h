//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_MAXIMUM_H
#define _CUDA_FUNCTIONAL_MAXIMUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif
#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif
#if _CCCL_HAS_FLOAT128()
#  include <crt/device_fp128_functions.h>
#endif

#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum
{
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
    noexcept(noexcept((__lhs < __rhs) ? __rhs : __lhs))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      if constexpr (_CUDA_VSTD::is_same_v<_Tp, float>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::fmaxf(__lhs, __rhs);))
      }
      else if constexpr (_CUDA_VSTD::is_same_v<_Tp, double>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return ::fmax(__lhs, __rhs);))
      }
#if _CCCL_HAS_NVFP16()
      else if constexpr (_CUDA_VSTD::is_same_v<_Tp, __half>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_53, (return ::__hmax(__lhs, __rhs);))
      }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
      else if constexpr (_CUDA_VSTD::is_same_v<_Tp, __nv_bfloat16>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_80, (return ::__hmax(__lhs, __rhs);))
      }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_FLOAT128()
      else if constexpr (_CUDA_VSTD::is_same_v<_Tp, __float128>)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_100, (return ::__nv_fp128_fmax(__lhs, __rhs);))
      }
#endif // _CCCL_HAS_FLOAT128()
    }
    return (__lhs < __rhs) ? __rhs : __lhs;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(maximum);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr _CUDA_VSTD::common_type_t<_T1, _T2>
  operator()(const _T1& __lhs, const _T2& __rhs) const noexcept(noexcept((__lhs < __rhs) ? __rhs : __lhs))
  {
    return (__lhs < __rhs) ? __rhs : __lhs;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MAXIMUM_H
