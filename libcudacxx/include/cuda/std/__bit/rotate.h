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
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotl_impl(_Tp __t, uint32_t __cnt_mod) noexcept
{
  return (__t << __cnt_mod) | (__t >> (numeric_limits<_Tp>::digits - __cnt_mod));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotr_impl(_Tp __t, uint32_t __cnt_mod) noexcept
{
  return (__t >> __cnt_mod) | (__t << (numeric_limits<_Tp>::digits - __cnt_mod));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotl(_Tp __t, uint32_t __cnt) noexcept
{
  static_assert(__cccl_is_unsigned_integer<_Tp>::value, "__rotl requires unsigned types");
  using __nlt = numeric_limits<_Tp>;
  if (!is_constant_evaluated() && sizeof(_Tp) <= sizeof(uint32_t))
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::__funnelshift_l(__t, __t, __cnt);), //
                      (return _CUDA_VSTD::__rotl_impl(__t, __cnt % __nlt::digits);))
  }
  return _CUDA_VSTD::__rotl_impl(__t, __cnt % __nlt::digits);
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotr(_Tp __t, uint32_t __cnt) noexcept
{
  static_assert(__cccl_is_unsigned_integer<_Tp>::value, "__rotr requires unsigned types");
  using __nlt = numeric_limits<_Tp>;
  if (!is_constant_evaluated() && sizeof(_Tp) <= sizeof(uint32_t))
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::__funnelshift_r(__t, __t, __cnt);), //
                      (return _CUDA_VSTD::__rotr_impl(__t, __cnt % __nlt::digits);))
  }
  return _CUDA_VSTD::__rotr_impl(__t, __cnt % __nlt::digits);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotl(_Tp __t, uint32_t __cnt) noexcept
{
  return _CUDA_VSTD::__rotl(__t, __cnt);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp rotr(_Tp __t, uint32_t __cnt) noexcept
{
  return _CUDA_VSTD::__rotr(__t, __cnt);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_ROTATE_H
