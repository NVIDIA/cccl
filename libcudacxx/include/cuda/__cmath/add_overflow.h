//===----------------------------------------------------------------------===//
//
// Part of libcu++, the _Common++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___CMATH_ADD_OVERFLOW_H
#define _CUDA___CMATH_ADD_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
struct overflow_check_result
{
  _Tp result;
  bool is_overflow;
};

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI overflow_check_result<decltype(_Ap{} + _Bp{})>
__add_overflow(const _Ap __a, const _Bp __b) noexcept
{
  using _Common = decltype(_Ap{} + _Bp{});
  if constexpr (sizeof(_Ap) < sizeof(int) && sizeof(_Bp) < sizeof(int))
  {
    return overflow_check_result<_Common>{__a + __b, false};
  }
  constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
  auto __a1              = static_cast<_Common>(__a);
  auto __b1              = static_cast<_Common>(__b);
  if constexpr (_CUDA_VSTD::is_unsigned_v<_Common>)
  {
    return overflow_check_result<_Common>{__a1 + __b1, (__a1 > __max_v - __b1)};
  }
  else
  {
    if ((__a1 <= 0 && __b1 >= 0) || (__a1 >= 0 && __b1 <= 0)) // opposite signs
    {
      return overflow_check_result<_Common>{__a1 + __b1, false};
    }
    // ^^^^ opposite signs ^^^^ / vvvv same signs vvvv
    constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
    if (__a1 == __min_v || __b1 == __min_v) // abs(min) is undefined behavior
    {
      return overflow_check_result<_Common>{__a1 + __b1, true};
    }
    auto __abs_a       = (__a1 < 0) ? -__a1 : __a1;
    auto __abs_b       = (__b1 < 0) ? -__b1 : __b1;
    bool __is_overflow = static_cast<_Common>(__abs_a) > __max_v - static_cast<_Common>(__abs_b);
    return overflow_check_result<_Common>{__a1 + __b1, __is_overflow};
  }
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI overflow_check_result<decltype(_Ap{} + _Bp{})>
__add_overflow_builtin(const _Ap __a, const _Bp __b) noexcept
{
  using _Common = decltype(_Ap{} + _Bp{});
  _Common __ret;
  bool __is_overflow = __builtin_add_overflow(__a, __b, &__ret);
  return overflow_check_result<_Common>{__ret, __is_overflow};
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI overflow_check_result<decltype(_Ap{} + _Bp{})>
add_overflow(const _Ap __a, const _Bp __b) noexcept
{
#if _CCCL_CHECK_BUILTIN(builtin_add_overflow)
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return ::cuda::__add_overflow(__a, __b);), //
                    (return ::cuda::__add_overflow_builtin(__a, __b);))
#else // ^^^^ _CCCL_CHECK_BUILTIN ^^^^ / vvvv !_CCCL_CHECK_BUILTIN vvvv
  return ::cuda::__add_overflow(__a, __b);
#endif // ^^^^ !_CCCL_CHECK_BUILTIN vvvv
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_ADD_OVERFLOW_H
