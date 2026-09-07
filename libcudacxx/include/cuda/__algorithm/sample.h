//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___ALGORITHM_SAMPLE
#define __CUDA___ALGORITHM_SAMPLE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/copy.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__random/uniform_real_distribution.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
//! @brief Selects `__n` elements from `__first[__index, __index + __N)` using Vitter's Method A.
//!
//! Method A draws one uniform variate per selected element and walks the remaining population
//! by whole gaps. It costs `O(__N)` arithmetic, but it avoids the rejection loop of Method D,
//! so it is faster when the sampling fraction is high.
//!
//! @param[in] __first Beginning of the population
//! @param[out] __output_iter Beginning of the destination range
//! @param[in,out] __index Index of the next candidate element, advanced past the selected elements
//! @param[in] __N_int Number of population elements that remain available
//! @param[in] __n_int Number of samples to draw, must satisfy `0 < __n <= __N`
//! @param[in,out] __g Uniform random number generator
//!
//! @return The end of the written destination range
template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
[[nodiscard]] _CCCL_API _SampleIterator __vitter_sample_method_a(
  _PopulationIterator __first,
  _SampleIterator __output_iter,
  _Distance __index,
  _Distance __N_int,
  _Distance __n_int,
  _UniformRandomNumberGenerator& __g)
{
  ::cuda::std::uniform_real_distribution<double> __uniform{};

  auto __top    = static_cast<double>(__N_int - __n_int);
  auto __N_real = static_cast<double>(__N_int);

  while (__n_int >= 2)
  {
    // `__top` counts the elements that are still available but will not be selected, and it
    // holds the invariant `__top == __N_real - __n`. When it reaches zero (or slightly less
    // than with floating point arithmetic), the remaining population is exactly the remaining
    // sample, so every element from here on is selected. The loop below would still find them,
    // but it would draw one discarded variate per element to do it.
    if (__top <= 0.0)
    {
      __output_iter = ::cuda::std::copy(__first + __index, __first + __index + __n_int, __output_iter);
      return __output_iter;
    }

    const auto __v = __uniform(__g);

    auto __s    = _Distance{0};
    auto __quot = __top / __N_real;

    while (__quot > __v)
    {
      ++__s;
      __top -= 1.0;
      __N_real -= 1.0;
      __quot = (__quot * __top) / __N_real;
    }

    // Skip over the next __s records and select the following one.
    __index += __s;
    *__output_iter++ = __first[__index++];

    __N_real -= 1.0;
    --__n_int;
  }

  // The `__n == 1` tail below draws a variate. When `__top` is zero (or slightly less due to
  // fp math) that variate cannot change the result, because `__N_real` is then 1 and the
  // product truncates to zero.
  if (__top <= 0.0)
  {
    *__output_iter++ = __first[__index++];
    return __output_iter;
  }

  // Special case __n == 1. The paper rounds Nreal first, then truncates the product, so that
  // roundoff can never produce the out-of-range value __s == __n_real.
  const auto __s = static_cast<_Distance>(::cuda::std::round(__N_real) * __uniform(__g));

  __index += __s;
  *__output_iter++ = __first[__index++];

  return __output_iter;
}

//! @brief Selects `__n` elements from `__first[0, __N)` using Vitter's Method D.
//!
//! @param[in] __first Beginning of the population
//! @param[out] __output_iter Beginning of the destination range
//! @param[in] __N_int Number of population elements
//! @param[in] __n_int Number of samples to draw, must satisfy `0 < __n <= __N`
//! @param[in,out] __g Uniform random number generator
//! @return The end of the written destination range
template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
[[nodiscard]] _CCCL_API _SampleIterator __vitter_sample_method_d(
  _PopulationIterator __first,
  _SampleIterator __output_iter,
  _Distance __N_int,
  _Distance __n_int,
  _UniformRandomNumberGenerator& __g)
{
  ::cuda::std::uniform_real_distribution<double> __uniform{};

  // Index of the next candidate element. Method D never moves it backwards, so the output is stable.
  auto __index = _Distance{0};

  auto __k      = __n_int;
  auto __n_real = static_cast<double>(__k);
  auto __n_inv  = 1.0 / __n_real;
  auto __N_real = static_cast<double>(__N_int);

  // V_prime is the __k-th root of a uniform variate. It carries across loop iterations, because a
  // rejected candidate still yields a usable variate for the next attempt.
  //
  // NOTE: The original algorithm uses exp(log(uniform(__g)) * __n_inv) but this just simplifies
  // to pow()
  auto __V_prime = ::cuda::std::pow(__uniform(__g), __n_inv);

  auto __qu1      = __N_int - __k + 1;
  auto __qu1_real = __N_real - __n_real + 1.0;

  // Ratio of remaining population to remaining samples below which Method A is faster.
  //
  // Method D stays in its acceptance-rejection loop while `__N_int > __SAMPLE_ALPHA_INV *
  // __n_int`, and hands the rest of the sample to Method A once the sampling fraction rises
  // above `1 / this value`. The paper calls the reciprocal `alpha` and stores `-1/alpha` in
  // the integer `negalphainv`. It uses `alpha = 1/13`, and it gives 0.05 to 0.15 as the useful
  // range for `alpha`, that is, `__SAMPLE_ALPHA_INV` between 7 and 20.
  constexpr int __SAMPLE_ALPHA_INV = 13;

  auto __threshold = __SAMPLE_ALPHA_INV * __k;

  while ((__k > 1) && (__threshold < __N_int))
  {
    const auto __n_min_1_inv = 1.0 / (__n_real - 1.0);

    auto __s          = _Distance{0};
    auto __neg_S_real = 0.0;

    do
    {
      // Step D2: generate U and X.
      double __x;

      do
      {
        __x = __N_real * (1.0 - __V_prime);
        __s = static_cast<_Distance>(__x);
        if (__s < __qu1)
        {
          break;
        }
        // NOTE: The original algorithm uses exp(log(uniform(__g)) * __n_inv) but this just
        // simplifies to pow()
        __V_prime = ::cuda::std::pow(__uniform(__g), __n_inv);
      } while (true);

      const auto __u = __uniform(__g);

      __neg_S_real = -static_cast<double>(__s);
      // Step D3: accept? This test succeeds with probability 1 - O(__k / __N).
      //
      // NOTE: The original algorithm uses exp(log(... * __n_min_1_inv) but this just simplifies
      // to pow()
      const auto __y1 = ::cuda::std::pow(__u * __N_real / __qu1_real, __n_min_1_inv);

      __V_prime = __y1 * ((-__x / __N_real) + 1.0) * (__qu1_real / (__neg_S_real + __qu1_real));
      if (__V_prime <= 1.0)
      {
        break;
      }

      // Step D4: accept? Evaluate the exact ratio with a bounded product.
      auto __y2     = 1.0;
      auto __top    = __N_real - 1.0;
      auto __bottom = 0.0;
      auto __limit  = 0.0;

      if (__k - 1 > __s)
      {
        __bottom = __N_real - __n_real;
        __limit  = __N_real + __neg_S_real;
      }
      else
      {
        __bottom = __neg_S_real + __N_real - 1.0;
        __limit  = static_cast<double>(__qu1);
      }

      // NOLINTNEXTLINE(bugprone-float-loop-counter)
      for (auto __t = __N_real - 1.0; __t >= __limit; __t -= 1.0, __top -= 1.0, __bottom -= 1.0)
      {
        __y2 = (__y2 * __top) / __bottom;
      }

      // Note: the paper has this as N/(-x + N) but we move to rhs and multiply because this is
      // more efficient on device. The compiler won't do this for us because these are all
      // floating point.
      //
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      if (__N_real >= (__N_real - __x) * __y1 * ::cuda::std::pow(__y2, __n_min_1_inv))
      {
        // Accept.
        __V_prime = ::cuda::std::pow(__uniform(__g), __n_min_1_inv);
        break;
      }

      __V_prime = ::cuda::std::pow(__uniform(__g), __n_inv);
    } while (true);

    // Step D5: select the (__s + 1)st record. Skip __s records, then take the next one.
    __index += __s;
    *__output_iter++ = __first[__index++];

    __N_int  = __N_int - __s - 1;
    __N_real = __neg_S_real + __N_real - 1.0;

    --__k;
    __n_real = __n_real - 1.0;
    __n_inv  = __n_min_1_inv;

    __qu1      = __qu1 - __s;
    __qu1_real = __neg_S_real + __qu1_real;

    __threshold -= __SAMPLE_ALPHA_INV;
  }

  if (__k > 1)
  {
    // The sampling fraction is now high enough that Method A is faster.
    return ::cuda::__detail::__vitter_sample_method_a(__first, __output_iter, __index, __N_int, __k, __g);
  }

  // Special case __k == 1. Reuse the carried Vprime rather than drawing again.
  const auto __s = static_cast<_Distance>(__N_real * __V_prime);

  __index += __s;
  *__output_iter++ = __first[__index];

  return __output_iter;
}
} // namespace __detail

//! @brief Selects `__n` elements from `[__first, __last)` without replacement, in population order.
//!
//! Implements Vitter's Method D, "An Efficient Algorithm for Sequential Random Sampling", ACM
//! Transactions on Mathematical Software, Vol. 13, No. 1, March 1987, pages 58-67
//! (https://www.ittc.ku.edu/~jsv/Papers/Vit87.RandomSampling.pdf).
//!
//! Unlike `cuda::std::sample`, which reads every population element, this algorithm reads exactly
//! the `min(__n, __last - __first)` selected elements and draws `O(__n)` random numbers. It requires
//! a random access population iterator so that skipped elements are never touched. Each selected
//! element is written in increasing population order, so the result is stable.
//!
//! The gap distribution is computed in `double`. The population size must therefore be exactly
//! representable as a `double`, that is, at most 2^53.
//!
//! @param[in] __first Beginning of the population
//! @param[in] __last End of the population
//! @param[out] __output_iter Beginning of the destination range
//! @param[in] __n Number of elements to select
//! @param[in,out] __g Uniform random number generator
//!
//! @return The end of the written destination range
template <class _PopulationIterator,
          class _PopulationSent,
          class _SampleIterator,
          class _Distance,
          class _UniformRandomNumberGenerator>
_CCCL_API _SampleIterator sample(
  _PopulationIterator __first,
  _PopulationSent __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator&& __g)
{
  static_assert(::cuda::std::__has_random_access_traversal<_PopulationIterator>,
                "PopulationIterator must meet the requirements of RandomAccessIterator");

  // We would use the usual ::cuda::std::common_type_t machinery, but clang-cuda 21+ crashes
  // when casting a __int128 to double, see https://github.com/llvm/llvm-project/issues/218919.
  using _CommonType = ::cuda::std::int64_t;

  if constexpr (::cuda::std::is_signed_v<_Distance>)
  {
    _CCCL_ASSERT(__n >= 0, "N must be a positive number.");
  }

  const auto __N = static_cast<_CommonType>(__last - __first);
  // __n might be UINT128T_MAX, which won't fit in our _CommonType so need safely clamp to __N
  // instead of just blindly casting min() to _CommonType.
  const auto __k = ::cuda::std::cmp_greater(__n, __N) ? __N : static_cast<_CommonType>(__n);

  if (__k <= 0)
  {
    return __output_iter;
  }

  if (__k >= __N)
  {
    return ::cuda::std::copy(__first, __last, __output_iter);
  }

  _CCCL_ASSERT(__N <= (_CommonType{1} << 53),
               "Population size must be exactly representable as a double, that is, at most 2^53.");

  return ::cuda::__detail::__vitter_sample_method_d(__first, __output_iter, __N, __k, __g);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___ALGORITHM_SAMPLE
