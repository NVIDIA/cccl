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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/detail/libcxx/include/__debug>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
template <class _Tp,
          class _Up,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_unsigned, _Tp), int> = 0,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp ceil_div(const _Tp __a, const _Up __b) noexcept
{
  _LIBCUDACXX_DEBUG_ASSERT(__b > _Tp(0), "cuda::ceil_div: b must be positive");
  using _UCommon   = _CUDA_VSTD::__make_unsigned_t<_CUDA_VSTD::__common_type_t<_Tp, _Up>>;
  const auto __res = static_cast<_UCommon>(__a) / static_cast<_UCommon>(__b);
  return static_cast<_Tp>(__res + (__res * static_cast<_UCommon>(__b) != static_cast<_UCommon>(__a)));
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
template <class _Tp,
          class _Up,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp), int>   = 0,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Up), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp ceil_div(const _Tp __a, const _Up __b) noexcept
{
  _LIBCUDACXX_DEBUG_ASSERT(__a >= _Tp(0), "cuda::ceil_div: a must be non negative");
  _LIBCUDACXX_DEBUG_ASSERT(__b > _Tp(0), "cuda::ceil_div: b must be positive");
  using _UCommon = _CUDA_VSTD::__make_unsigned_t<_CUDA_VSTD::__common_type_t<_Tp, _Up>>;
  // Due to the precondition `__a >= 0` we can safely cast to unsigned without danger of overflowing
  return static_cast<_Tp>((static_cast<_UCommon>(__a) + static_cast<_UCommon>(__b) - 1) / static_cast<_UCommon>(__b));
}

//! @brief Divides two numbers \p __a and \p __b, rounding up if there is a remainder, \p __b is an enum
//! @param __a The dividend
//! @param __b The divisor
//! @pre \p __a must be non-negative
//! @pre \p __b must be positive
template <class _Tp,
          class _Up,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_integral, _Tp), int> = 0,
          _CUDA_VSTD::__enable_if_t<_CCCL_TRAIT(_CUDA_VSTD::is_enum, _Up), int>     = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp ceil_div(const _Tp __a, const _Up __b) noexcept
{
  return ::cuda::ceil_div(__a, static_cast<_CUDA_VSTD::__underlying_type_t<_Up>>(__b));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_CEIL_DIV_H
