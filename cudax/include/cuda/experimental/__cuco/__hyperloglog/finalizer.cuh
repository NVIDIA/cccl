//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO__HYPERLOGLOG__FINALIZER_CUH
#define _CUDAX__CUCO__HYPERLOGLOG__FINALIZER_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/experimental/__cuco/__hyperloglog/tuning.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__hyperloglog_ns
{
//! @brief Estimate correction algorithm based on HyperLogLog++.
//!
//! @note Variable names correspond to the definitions given in the HLL++ paper:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf
//! @note Previcion must be >= 4.
//!
class _Finalizer
{
  // Note: Most of the types in this implementation are explicit instead of relying on `auto` to
  // avoid confusion with the reference implementation.

public:
  //! @brief Constructs an HLL finalizer object.
  //!
  //! @param precision HLL precision parameter
  _CCCL_API constexpr _Finalizer(int __precision_)
      : __precision{__precision_}
      , __m{static_cast<int>(1u << __precision_)}
  {}

  //! @brief Compute the bias-corrected cardinality estimate.
  //!
  //! @param z Geometric mean of registers
  //! @param v Number of 0 registers
  //!
  //! @return Bias-corrected cardinality estimate
  [[nodiscard]] _CCCL_API ::cuda::std::size_t operator()(double __z, int __v) const noexcept
  {
    double __e = __alpha_mm() / __z;

    if (__v > 0)
    {
      // Use linear counting for small cardinality estimates.
      const double __h = __m * ::cuda::std::log(static_cast<double>(__m) / __v);
      // The threshold `2.5 * m` is from the original HLL algorithm.
      if (__e <= 2.5 * __m)
      {
        return static_cast<::cuda::std::size_t>(::cuda::std::round(__h));
      }

      if (__precision < 19)
      {
        __e = (__h <= __hyperloglog_ns::__threshold(__precision)) ? __h : __bias_corrected_estimate(__e);
      }
    }
    else
    {
      // HLL++ is defined only when p < 19, otherwise we need to fallback to HLL.
      if (__precision < 19)
      {
        __e = __bias_corrected_estimate(__e);
      }
    }

    return static_cast<::cuda::std::size_t>(::cuda::std::round(__e));
  }

private:
  [[nodiscard]] _CCCL_API constexpr double __alpha_mm() const noexcept
  {
    switch (__m)
    {
      case 16:
        return 0.673 * __m * __m;
      case 32:
        return 0.697 * __m * __m;
      case 64:
        return 0.709 * __m * __m;
      default:
        return (0.7213 / (1.0 + 1.079 / __m)) * __m * __m;
    }
  }

  [[nodiscard]] _CCCL_API constexpr double __bias_corrected_estimate(double __e) const noexcept
  {
    return (__e < 5.0 * __m) ? __e - __bias(__e) : __e;
  }

  [[nodiscard]] _CCCL_API constexpr double __bias(double __e) const noexcept
  {
    const auto __anchor_index = __interpolation_anchor_index(__e);
    const int __n             = __raw_estimate_data_size(__precision);

    auto __low  = ::cuda::std::max(__anchor_index - __k + 1, 0);
    auto __high = ::cuda::std::min(__low + __k, __n);
    // Keep moving bounds as long as the (exclusive) high bound is closer to the estimate than
    // the lower (inclusive) bound.
    while (__high < __n and __distance(__e, __high) < __distance(__e, __low))
    {
      __low += 1;
      __high += 1;
    }

    auto __biases     = __bias_data(__precision);
    double __bias_sum = 0.0;
    for (int __i = __low; __i < __high; ++__i)
    {
      __bias_sum += __biases[__i];
    }

    return __bias_sum / (__high - __low);
  }

  [[nodiscard]] _CCCL_API constexpr double __distance(double __e, int __i) const noexcept
  {
    const auto __diff = __e - __raw_estimate_data(__precision)[__i];
    return __diff * __diff;
  }

  [[nodiscard]] _CCCL_API constexpr int __interpolation_anchor_index(double __e) const noexcept
  {
    auto __estimates      = __raw_estimate_data(__precision);
    const int __n         = __raw_estimate_data_size(__precision);
    int __left            = 0;
    int __right           = static_cast<int>(__n) - 1;
    int __mid             = -1;
    int __candidate_index = 0; // Index of the closest element found

    while (__left <= __right)
    {
      __mid = __left + (__right - __left) / 2;

      if (__estimates[__mid] < __e)
      {
        __left = __mid + 1;
      }
      else if (__estimates[__mid] > __e)
      {
        __right = __mid - 1;
      }
      else
      {
        // Exact match found, no need to look further
        return __mid;
      }
    }

    // At this point, '__left' is the insertion point. We need to compare the elements at '__left' and
    // '__left - 1' to find the closest one, taking care of boundary conditions.

    // Distance from '__e' to the element at '__left', if within bounds
    const double __dist_lhs =
      __left < static_cast<int>(__n)
        ? ::cuda::std::abs(__estimates[__left] - __e)
        : ::cuda::std::numeric_limits<double>::max();
    // Distance from '__e' to the element at '__left - 1', if within bounds
    const double __dist_rhs =
      __left - 1 >= 0 ? ::cuda::std::abs(__estimates[__left - 1] - __e) : ::cuda::std::numeric_limits<double>::max();

    __candidate_index = (__dist_lhs < __dist_rhs) ? __left : __left - 1;

    return __candidate_index;
  }

  static constexpr auto __k = 6; ///< Number of interpolation points to consider
  int __precision; ///< HLL precision parameter
  int __m; ///< Number of registers (2^precision)
};
} // namespace cuda::experimental::cuco::__hyperloglog_ns

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO__HYPERLOGLOG__FINALIZER_CUH
