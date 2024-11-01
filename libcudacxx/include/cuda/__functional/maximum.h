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

#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__type_traits/common_type.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum : __binary_function<_Tp, _Tp, _Tp>
{
  using __result_type = _Tp; // used by valarray

  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
    noexcept(noexcept(__x < __y ? __y : __x))
  {
    return (__x < __y) ? __y : __x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(maximum);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT maximum<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI typename _CUDA_VSTD::common_type<_T1, _T2>::type
  operator()(const _T1& __t, const _T2& __u) const noexcept(noexcept((__t < __u) ? __u : __t))
  {
    return (__t < __u) ? __u : __t;
  }

  using is_transparent = void;
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA_FUNCTIONAL_MAXIMUM_H
