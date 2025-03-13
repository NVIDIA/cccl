//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ILOG
#define _LIBCUDACXX___CMATH_ILOG

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
_LIBCUDACXX_HIDE_FROM_ABI constexpr int ilog2(_Tp __t) noexcept
{
  using _Up = _CUDA_VSTD::make_unsigned_t<_Tp>;
  _CCCL_ASSERT(__t > 0, "ilog2() argument must be strictly positive");
  auto __log10_approx = _CUDA_VSTD::__bit_log2(static_cast<_Up>(__t));
  _CCCL_ASSUME(__log10_approx <= _CUDA_VSTD::numeric_limits<_Tp>::digits);
  return __log10_approx;
}

static _CCCL_GLOBAL_CONSTANT uint32_t __power_of_10_32bit[] = {
  1, 10, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 100'000'000, 1'000'000'000};

static _CCCL_GLOBAL_CONSTANT uint64_t __power_of_10_64bit[] = {
  1,
  10,
  100,
  1'000,
  10'000,
  100'000,
  1'000'000,
  10'000'000,
  100'000'000,
  1'000'000'000,
  10'000'000'000,
  100'000'000'000,
  1'000'000'000'000,
  10'000'000'000'000,
  100'000'000'000'000,
  1'000'000'000'000'000,
  10'000'000'000'000'000,
  100'000'000'000'000'000,
  1'000'000'000'000'000'000,
  10'000'000'000'000'000'000ull};

#if _CCCL_HAS_INT128()

static _CCCL_GLOBAL_CONSTANT __uint128_t __power_of_10_128bit[] = {
  1,
  10,
  100,
  1'000,
  10'000,
  100'000,
  1'000'000,
  10'000'000,
  100'000'000,
  1'000'000'000,
  10'000'000'000,
  100'000'000'000,
  1'000'000'000'000,
  10'000'000'000'000,
  100'000'000'000'000,
  1'000'000'000'000'000,
  10'000'000'000'000'000,
  100'000'000'000'000'000,
  1'000'000'000'000'000'000,
  10'000'000'000'000'000'000ull,
  __uint128_t{10'000'000'000'000'000'000ull} * 10,
  __uint128_t{10'000'000'000'000'000'000ull} * 100,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 10'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 100'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 10'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 100'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 10'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 100'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 10'000'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 100'000'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000'000'000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000'000'0000,
  __uint128_t{10'000'000'000'000'000'000ull} * 10'000'000'000'000'0000,
  __uint128_t{10'000'000'000'000'000'000ull} * 100'000'000'000'000'0000,
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000'000'000'0000ull};

#endif // _CCCL_HAS_INT128()

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Tp))
_LIBCUDACXX_HIDE_FROM_ABI constexpr int ilog10(_Tp __t) noexcept
{
  _CCCL_ASSERT(__t > 0, "ilog10() argument must be strictly positive");
  constexpr auto __reciprocal_log2_10 = 1.0f / 3.321928094f; // 1 / log2(10)
  auto __log2                         = ::cuda::ilog2(__t) * __reciprocal_log2_10;
  auto __log10_f      = _CUDA_VSTD::__cccl_default_is_constant_evaluated() ? __log2 + 0.5f : _CUDA_VSTD::ceil(__log2);
  auto __log10_approx = static_cast<int>(__log10_f);
  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    __log10_approx -= static_cast<uint32_t>(__t) < __power_of_10_32bit[__log10_approx];
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    __log10_approx -= static_cast<uint64_t>(__t) < __power_of_10_64bit[__log10_approx];
  }
#if _CCCL_HAS_INT128()
  else
  {
    __log10_approx -= static_cast<__uint128_t>(__t) < __power_of_10_128bit[__log10_approx];
  }
#endif // _CCCL_HAS_INT128()
  _CCCL_ASSUME(__log10_approx <= _CUDA_VSTD::numeric_limits<_Tp>::digits / 3); // 2^X < 10^(x/3) -> 8^X < 10^x
  return __log10_approx;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CMATH_ILOG
