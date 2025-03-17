//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ROTATE_H
#define _LIBCUDACXX___BIT_ROTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotr(_Tp __t, int __cnt) noexcept
{
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  if constexpr (is_same_v<_Tp, uint32_t>)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_r(__t, __t, __cnt);))
    }
  }
  auto __cnt_mod = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __t : (__t >> __cnt_mod) | (__t << (__digits - __cnt_mod));
}

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotl(_Tp __t, int __cnt) noexcept
{
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  if constexpr (is_same_v<_Tp, uint32_t>)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_l(__t, __t, __cnt);))
    }
  }
  auto __cnt_mod = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __t : (__t << __cnt_mod) | (__t >> (__digits - __cnt_mod));
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotl(_Tp __t, int __cnt) noexcept
{
  if (__cnt < 0)
  {
    _CCCL_ASSERT(__cnt != numeric_limits<int>::min(), "__cnt overflow");
    return _CUDA_VSTD::__rotr(__t, -__cnt);
  }
  return _CUDA_VSTD::__rotl(__t, __cnt);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotr(_Tp __t, int __cnt) noexcept
{
  if (__cnt < 0)
  {
    _CCCL_ASSERT(__cnt != numeric_limits<int>::min(), "__cnt overflow");
    return _CUDA_VSTD::__rotl(__t, -__cnt);
  }
  return _CUDA_VSTD::__rotr(__t, __cnt);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_ROTATE_H
