//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_ILOG_H
#define _CUDA___CMATH_ILOG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_cv_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ilog2(const _Tp __t) noexcept
{
  using _Up = ::cuda::std::make_unsigned_t<_Tp>;
  _CCCL_ASSERT(__t > 0, "ilog2() argument must be strictly positive");
  auto __log2_approx = ::cuda::std::__bit_log2(static_cast<_Up>(__t));
  _CCCL_ASSUME(__log2_approx <= ::cuda::std::numeric_limits<_Tp>::digits);
  return __log2_approx;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_cv_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ceil_ilog2(const _Tp __t) noexcept
{
  using _Up = ::cuda::std::make_unsigned_t<_Tp>;
  return ::cuda::ilog2(__t) + !::cuda::std::has_single_bit(static_cast<_Up>(__t));
}

// Construct a representable power of ten without a memory lookup. The low three exponent bits
// produce a 32-bit factor of at most 10^7. Use constant multiplications for 64-bit results and select
// the remaining factor before multiplying for 128-bit results.
template <class _Up>
[[nodiscard]] _CCCL_API constexpr _Up __ilog10_power_of_10(unsigned __exponent) noexcept
{
  ::cuda::std::uint32_t __low = (__exponent & 1) ? 10u : 1u;
  __low                       = (__exponent & 2) ? __low * 100u : __low;
  __low                       = (__exponent & 4) ? __low * 10'000u : __low;
  if constexpr (sizeof(_Up) <= sizeof(::cuda::std::uint32_t))
  {
    return (__exponent & 8) ? __low * 100'000'000u : __low;
  }
  else if constexpr (sizeof(_Up) == sizeof(::cuda::std::uint64_t))
  {
    const auto __power = (__exponent & 8) ? static_cast<_Up>(__low) * 100'000'000u : static_cast<_Up>(__low);
    return (__exponent & 16) ? __power * 10'000'000'000'000'000ull : __power;
  }
  else
  {
    _Up __high                 = (__exponent & 8) ? _Up{100'000'000} : _Up{1};
    constexpr auto __power16   = _Up{10'000'000'000'000'000ull};
    const auto __power16_or_24 = (__exponent & 8) ? __power16 * 100'000'000u : __power16;
    __high                     = (__exponent & 16) ? __power16_or_24 : __high;
    __high                     = (__exponent & 32) ? __power16 * __power16 : __high;
    return static_cast<_Up>(__low) * __high;
  }
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_cv_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ilog10(const _Tp __t) noexcept
{
  _CCCL_ASSERT(__t > 0, "cuda::ilog10() argument must be strictly positive");
  if constexpr (sizeof(_Tp) <= 2)
  {
    int __result = (__t >= 10) + (__t >= 100);
    if constexpr (sizeof(_Tp) == 2)
    {
      __result += (__t >= 1'000) + (__t >= 10'000);
    }
    return __result;
  }
  else
  {
    using _Up = ::cuda::std::make_unsigned_t<::cuda::std::remove_cv_t<_Tp>>;
    // For bit widths b in [1, 128], floor(b * log10(2)) == (b * 1233) >> 12.
    // This is either ilog10(t) or one greater, so one exact comparison corrects the estimate.
    const auto __bit_width = static_cast<unsigned>(::cuda::ilog2(__t) + 1);
    const auto __estimate  = (__bit_width * 1233u) >> 12;
    const auto __power     = ::cuda::__ilog10_power_of_10<_Up>(__estimate);
    const auto __result    = static_cast<int>(__estimate) - (static_cast<_Up>(__t) < __power);
    _CCCL_ASSUME(__result <= ::cuda::std::numeric_limits<_Tp>::digits / 3); // 2^X < 10^(X/3)
    return __result;
  }
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_cv_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ceil_ilog10(const _Tp __t) noexcept
{
  _CCCL_ASSERT(__t > 0, "cuda::ceil_ilog10() argument must be strictly positive");
  return __t == 1 ? 0 : ::cuda::ilog10(static_cast<_Tp>(__t - 1)) + 1;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_ILOG_H
