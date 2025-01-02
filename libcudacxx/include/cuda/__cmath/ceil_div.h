//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_CEIL_DIV_H
#define _CUDA___CMATH_CEIL_DIV_H

d
#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/underlying_type.h>

  _LIBCUDACXX_BEGIN_NAMESPACE_CUDA

  template <class _UCommon>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _UCommon
  __unsigned_ceil_div(const _UCommon __a, const _UCommon __b) noexcept
{
  const auto __res = __a / __b;
  return __res + (__res * __b != __a);
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
template <class _Tp,
          class _Up,
          _CUDA_VSTD::enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp), int> = 0,
          _CUDA_VSTD::enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 decltype(_Tp{} / _Up{})
ceil_div(const _Tp __a, const _Up __b) noexcept
{
  _CCCL_ASSERT(__b > _Up{0}, "cuda::ceil_div: b must be positive");
  using _Common  = decltype(_Tp{} / _Up{});
  using _UCommon = _CUDA_VSTD::make_unsigned_t<_Common>;
  if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__a >= _Tp{0}, "cuda::ceil_div: a must be non negative");
  }
  auto __a1 = static_cast<_UCommon>(__a);
  auto __b1 = static_cast<_UCommon>(__b);
  if constexpr (_CUDA_VSTD::is_signed_v<_Common>)
  {
    return static_cast<_Common>((__a1 + __b1 - 1) / __b1);
  }
  else
  {
    // the ::min method is faster even if __b is a compile-time constant
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return static_cast<_Common>(_CUDA_VSTD::min(__a1, 1 + ((__a1 - 1) / __b1)));),
                      (const auto __res = __a1 / __b1; //
                       return static_cast<_Common>(__res + (__res * __b1 != __a1));))
  }
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
template <class _Tp,
          class _Up,
          _CUDA_VSTD::enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp), int> = 0,
          _CUDA_VSTD::enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up), int>     = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(__a, static_cast<_CUDA_VSTD::underlying_type_t<_Up>>(__b));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_CEIL_DIV_H
