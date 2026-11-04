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
#include <cuda/std/__iterator/size.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <nv/target>

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

template <class _Tp, class = void>
_CCCL_GLOBAL_CONSTANT ::cuda::std::enable_if_t<false, _Tp> __powers_of_10{};

template <class _Tp>
_CCCL_GLOBAL_CONSTANT ::cuda::std::uint32_t
  __powers_of_10<_Tp, ::cuda::std::enable_if_t<sizeof(_Tp) == sizeof(::cuda::std::uint32_t)>>[10]{
    10,
    100,
    1'000,
    10'000,
    100'000,
    1'000'000,
    10'000'000,
    100'000'000,
    1'000'000'000,
    ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max()};

template <class _Tp>
_CCCL_GLOBAL_CONSTANT ::cuda::std::uint64_t
  __powers_of_10<_Tp, ::cuda::std::enable_if_t<sizeof(_Tp) == sizeof(::cuda::std::uint64_t)>>[20]{
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
    ::cuda::std::numeric_limits<::cuda::std::uint64_t>::max()};

#if _CCCL_HAS_INT128()
template <class _Tp>
_CCCL_GLOBAL_CONSTANT __uint128_t __powers_of_10<_Tp, ::cuda::std::enable_if_t<sizeof(_Tp) == sizeof(__uint128_t)>>[39]{
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
  __uint128_t{10'000'000'000'000'000'000ull} * 1'000'000'000'000'000'0000ull,
  ::cuda::std::numeric_limits<__uint128_t>::max()};
#endif // _CCCL_HAS_INT128()

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __ilog10_generic(const _Tp __v) noexcept
{
  using _Up                           = ::cuda::std::make_unsigned_t<_Tp>;
  constexpr auto __reciprocal_log2_10 = 0.301029995663f; // 1 / log2(10)
  constexpr const auto& __pows_of_10  = __powers_of_10<_Up>;

  const auto __log2   = ::cuda::ilog2(__v) * __reciprocal_log2_10;
  auto __log10_approx = static_cast<int>(__log2);

  _CCCL_ASSERT(__log10_approx < static_cast<int>(::cuda::std::size(__pows_of_10)), "out of bounds");

  if constexpr (::cuda::std::is_unsigned_v<_Tp> && sizeof(_Tp) != sizeof(::cuda::std::uint64_t))
  {
    // don't replace +1 with >= because wraparound behavior is needed here
    __log10_approx += static_cast<_Up>(__v) + 1 > __pows_of_10[__log10_approx];
  }
  else
  {
    __log10_approx += static_cast<_Up>(__v) >= __pows_of_10[__log10_approx];
  }
  return __log10_approx;
}

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
// Construct a representable power of ten without a memory lookup. The low three exponent bits
// produce a 32-bit factor of at most 10^7. Use constant multiplications for 64-bit results and select
// the remaining factor before multiplying for 128-bit results.
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __ilog10_device_power_of_10(unsigned __exponent) noexcept
{
  using ::cuda::std::uint32_t;
  uint32_t __low = (__exponent & 1) ? 10u : 1u;
  __low          = (__exponent & 2) ? __low * 100u : __low;
  __low          = (__exponent & 4) ? __low * 10'000u : __low;

  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return (__exponent & 8) ? __low * 100'000'000u : __low;
  }
  else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint64_t))
  {
    const auto __power = (__exponent & 8) ? static_cast<_Tp>(__low) * 100'000'000u : static_cast<_Tp>(__low);
    return (__exponent & 16) ? __power * 10'000'000'000'000'000ull : __power;
  }
  else
  {
    _Tp __high                 = (__exponent & 8) ? _Tp{100'000'000} : _Tp{1};
    constexpr auto __power16   = _Tp{10'000'000'000'000'000ull};
    const auto __power16_or_24 = (__exponent & 8) ? __power16 * 100'000'000u : __power16;
    __high                     = (__exponent & 16) ? __power16_or_24 : __high;
    __high                     = (__exponent & 32) ? __power16 * __power16 : __high;
    return static_cast<_Tp>(__low) * __high;
  }
}

// Fuse the exact correction into the subtraction: subc subtracts one from the estimate if value < power.
template <class _Tp>
[[nodiscard]] _CCCL_CONST _CCCL_DEVICE_API int
__ilog10_device_correct(_Tp __value, _Tp __power, unsigned __estimate) noexcept
{
  using ::cuda::std::uint64_t;
  int __result;
  // NOLINTBEGIN(bugprone-branch-clone): The PTX widths and register constraints differ.
  if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint32_t))
  {
    asm("{ .reg .u32 tmp;"
        "sub.cc.u32 tmp, %1, %2;"
        "subc.u32 %0, %3, 0; }"
        : "=r"(__result)
        : "r"(__value), "r"(__power), "r"(__estimate));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    asm("{ .reg .u64 tmp;"
        "sub.cc.u64 tmp, %1, %2;"
        "subc.u32 %0, %3, 0; }"
        : "=r"(__result)
        : "l"(__value), "l"(__power), "r"(__estimate));
  }
  else
  {
    asm("{ .reg .u64 tmp;"
        "sub.cc.u64 tmp, %1, %3;"
        "subc.cc.u64 tmp, %2, %4;"
        "subc.u32 %0, %5, 0; }"
        : "=r"(__result)
        : "l"(static_cast<uint64_t>(__value)),
          "l"(static_cast<uint64_t>(__value >> 64)),
          "l"(static_cast<uint64_t>(__power)),
          "l"(static_cast<uint64_t>(__power >> 64)),
          "r"(__estimate));
  }
  // NOLINTEND(bugprone-branch-clone)
  return __result;
}

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API int __ilog10_device(const _Tp __v) noexcept
{
  // For bit widths b in [1, 128], floor(b * log10(2)) == (b * 1233) >> 12.
  // This is either ilog10(v) or one greater, so one exact comparison corrects the estimate.
  // Counting leading zeros directly avoids selecting a 32-bit word before the 64-bit highest-bit query.
  const auto __bit_width = static_cast<unsigned>(::cuda::ilog2(__v) + 1);
  const auto __estimate  = (__bit_width * 1233u) >> 12;
  const auto __power     = ::cuda::__ilog10_device_power_of_10<_Tp>(__estimate);
  return ::cuda::__ilog10_device_correct(__v, __power, __estimate);
}

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_cv_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int ilog10(const _Tp __v) noexcept
{
  _CCCL_ASSERT(__v > 0, "cuda::ilog10() argument must be strictly positive");
  using _Up       = ::cuda::std::make_unsigned_t<_Tp>;
  const auto __uv = static_cast<_Up>(__v);

  if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint8_t))
  {
    return (__uv >= 10) + (__uv >= 100);
  }
  else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint16_t))
  {
    return (__uv >= 10) + (__uv >= 100) + (__uv >= 1'000) + (__uv >= 10'000);
  }
  else
  {
    int __ret = 0;
#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_ELSE_TARGET(NV_IS_DEVICE, (__ret = ::cuda::__ilog10_device(__uv);), (__ret = ::cuda::__ilog10_generic(__v);))
    }
    else
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
    {
      __ret = ::cuda::__ilog10_generic(__v);
    }
    _CCCL_ASSUME(__ret >= 0 && __ret <= ::cuda::std::numeric_limits<_Tp>::digits / 3); // 2^X < 10^(X/3)
    return __ret;
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
