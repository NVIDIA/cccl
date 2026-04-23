//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_PRIME_HPP
#define _CUDAX___CUCO___DETAIL_PRIME_HPP

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Modular multiplication: `(__n1 * __n2) % __m` without overflow.
[[nodiscard]] _CCCL_API constexpr ::cuda::std::uint64_t
__mod_mul(::cuda::std::uint64_t __n1, ::cuda::std::uint64_t __n2, ::cuda::std::uint64_t __m) noexcept
{
#if _CCCL_HAS_INT128()
  auto __r = static_cast<unsigned __int128>(__n1) * __n2;
  return static_cast<::cuda::std::uint64_t>(__r % __m);
#else
  // Fallback: Russian-peasant multiplication in modular arithmetic.
  ::cuda::std::uint64_t __r = 0;
  __n1 %= __m;
  __n2 %= __m;
  while (__n2 > 0)
  {
    if (__n2 & 1)
    {
      __r = (__r >= __m - __n1) ? __r - (__m - __n1) : __r + __n1;
    }
    __n1 = (__n1 >= __m - __n1) ? __n1 - (__m - __n1) : __n1 + __n1;
    __n2 >>= 1;
  }
  return __r;
#endif
}

//! @brief Modular exponentiation: `(__b ^ __e) % __m` via binary exponentiation.
[[nodiscard]] _CCCL_API constexpr ::cuda::std::uint64_t
__mod_pow(::cuda::std::uint64_t __b, ::cuda::std::uint64_t __e, ::cuda::std::uint64_t __m) noexcept
{
  ::cuda::std::uint64_t __r = 1;
  __b %= __m;
  while (__e > 0)
  {
    if (__e & 1)
    {
      __r = __mod_mul(__r, __b, __m);
    }
    __b = __mod_mul(__b, __b, __m);
    __e >>= 1;
  }
  return __r;
}

//! @brief Single Miller-Rabin witness test.
//!
//! Given `__n - 1 == 2^__s * __d`, checks whether `__a^__d == 1 (mod __n)` or
//! `__a^(2^__r * __d) == __n - 1 (mod __n)` for some `0 <= __r < __s`.
[[nodiscard]] _CCCL_API constexpr bool __miller_rabin_test(
  ::cuda::std::uint64_t __n, ::cuda::std::uint64_t __a, ::cuda::std::uint64_t __d, ::cuda::std::uint32_t __s) noexcept
{
  ::cuda::std::uint64_t __x             = __mod_pow(__a % __n, __d, __n);
  ::cuda::std::uint64_t const __neg_one = __n - 1;
  if (__x == 1 || __x == __neg_one)
  {
    return true;
  }

  for (::cuda::std::uint32_t __i = 1; __i < __s; ++__i)
  {
    __x = __mod_mul(__x, __x, __n);
    if (__x == __neg_one)
    {
      return true;
    }
  }
  return false;
}

//! @brief Deterministic primality test for all 64-bit integers.
//!
//! Uses trial division by small primes followed by Miller-Rabin with a fixed
//! set of bases that make the test deterministic for every `uint64_t`.
//! Bases from https://cp-algorithms.com/algebra/primality_tests.html.
[[nodiscard]] _CCCL_API constexpr bool __is_prime(::cuda::std::uint64_t __n) noexcept
{
  if (__n < 2)
  {
    return false;
  }

  // Trial division by small primes.
  for (::cuda::std::uint64_t __p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull})
  {
    if (__n % __p == 0)
    {
      return __n == __p;
    }
  }

  // Decompose `__n - 1 == 2^__s * __d`.
  ::cuda::std::uint64_t __d = __n - 1;
  ::cuda::std::uint32_t __s = 0;
  while ((__d & 1) == 0)
  {
    __d >>= 1;
    ++__s;
  }

  // Deterministic witness bases for all `uint64_t` values.
  for (::cuda::std::uint64_t __a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull})
  {
    if (!__miller_rabin_test(__n, __a, __d, __s))
    {
      return false;
    }
  }

  return true;
}

//! @brief Returns the smallest prime `>= __n`.
//!
//! For `__n <= 2`, returns 2. Otherwise searches odd numbers starting from
//! `__n` (or `__n + 1` if `__n` is even).
[[nodiscard]] _CCCL_API constexpr ::cuda::std::uint64_t __next_prime(::cuda::std::uint64_t __n) noexcept
{
  if (__n <= 2ull)
  {
    return 2ull;
  }

  __n |= 1ull; // make odd

  while (!__is_prime(__n))
  {
    __n += 2ull;
  }

  return __n;
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_PRIME_HPP
