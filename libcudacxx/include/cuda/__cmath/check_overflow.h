//===----------------------------------------------------------------------===//
//
// Part of libcu++, the _Common++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___CMATH_IS_OVERFLOW_H
#define _CUDA___CMATH_IS_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_add_overflow(const _Ap __a, const _Bp __b) noexcept
{
  using _Common          = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
  constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
  auto __a1              = static_cast<_Common>(__a);
  auto __b1              = static_cast<_Common>(__b);
  return (_CUDA_VSTD::is_unsigned_v<_Common> || __b1 >= 0) ? (__a1 > __max_v - __b1) : (__a1 < __min_v - __b1);
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_sub_overflow(const _Ap __a, const _Bp __b) noexcept
{
  return ::cuda::is_add_overflow(__a, -__b);
}

_CCCL_TEMPLATE(class _Ap, class _Bp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Ap) _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::is_integral, _Bp))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_mul_overflow(const _Ap __a, const _Bp __b) noexcept
{
  using _Common          = _CUDA_VSTD::common_type_t<_Ap, _Bp>;
  constexpr auto __max_v = _CUDA_VSTD::numeric_limits<_Common>::max();
  constexpr auto __min_v = _CUDA_VSTD::numeric_limits<_Common>::min();
  auto __a1              = static_cast<_Common>(__a);
  auto __b1              = static_cast<_Common>(__b);
  if (__b1 == 0)
  {
    return false;
  }
  auto __same_sign = (__a1 ^ __b1) >= 0;
  return (_CUDA_VSTD::is_unsigned_v<_Common> || __same_sign) ? (__a1 > __max_v / __b1) : (__a1 < __min_v / __b1);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_IS_OVERFLOW_H
