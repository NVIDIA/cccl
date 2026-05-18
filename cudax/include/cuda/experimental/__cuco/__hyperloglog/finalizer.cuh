//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___HYPERLOGLOG_FINALIZER_CUH
#define _CUDAX___CUCO___HYPERLOGLOG_FINALIZER_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__numeric/midpoint.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/__hyperloglog/tuning.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__hyperloglog_ns
{
//! @brief Estimate correction algorithm based on HyperLogLog++.
//!
//! @note Variable names correspond to the definitions given in the HLL++ paper:
//! https://static.googleusercontent.com/media/research.google.com/de//pubs/archive/40671.pdf
//! @note Precision must be >= 4.
//!
class hllpp_finalizer
{
  // Note: Most of the types in this implementation are explicit instead of relying on `auto` to
  // avoid confusion with the reference implementation.

public:
  //! @brief Constructs an HLL finalizer object.
  //!
  //! @param __precision_ HLL precision parameter
  _CCCL_HOST_DEVICE_API constexpr hllpp_finalizer(::cuda::std::int32_t __precision_) noexcept
      : __precision{__precision_}
      , __m{static_cast<::cuda::std::int32_t>(1u << __precision_)}
  {
    _CCCL_ASSERT(::cuda::in_range(__precision_, 4, 18), "Precision must be between 4 and 18");
  }

  //! @brief Compute the bias-corrected cardinality estimate.
  //!
  //! @param __z Geometric mean of registers
  //! @param __v Number of 0 registers
  //!
  //! @return Bias-corrected cardinality estimate
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::size_t operator()(double __z, ::cuda::std::int32_t __v) const noexcept
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
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double __alpha_mm() const noexcept
  {
    const auto __m2 = static_cast<double>(__m) * __m;
    switch (__m)
    {
      case 16:
        return 0.673 * __m2;
      case 32:
        return 0.697 * __m2;
      case 64:
        return 0.709 * __m2;
      default:
        return (0.7213 / (1.0 + 1.079 / __m)) * __m2;
    }
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double __bias_corrected_estimate(double __e) const noexcept
  {
    return (__e < 5.0 * __m) ? __e - __bias(__e) : __e;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double __bias(double __e) const noexcept
  {
    const auto __anchor_index = __interpolation_anchor_index(__e);
    const auto __n = static_cast<::cuda::std::int32_t>(__hyperloglog_ns::__raw_estimate_data_size(__precision));

    auto __low  = ::cuda::std::max(__anchor_index - __k + 1, ::cuda::std::int32_t{0});
    auto __high = ::cuda::std::min(__low + __k, __n);
    // Keep moving bounds as long as the (exclusive) high bound is closer to the estimate than
    // the lower (inclusive) bound.
    while (__high < __n && this->__distance(__e, __high) < this->__distance(__e, __low))
    {
      __low += 1;
      __high += 1;
    }

    const auto __biases = __hyperloglog_ns::__bias_data(__precision);
    double __bias_sum   = 0.0;
    for (::cuda::std::int32_t __i = __low; __i < __high; ++__i)
    {
      __bias_sum += __biases[__i];
    }

    return __bias_sum / (__high - __low);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr double __distance(double __e, ::cuda::std::int32_t __i) const noexcept
  {
    const auto __diff = __e - __hyperloglog_ns::__raw_estimate_data(__precision)[__i];
    return __diff * __diff;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::int32_t
  __interpolation_anchor_index(double __e) const noexcept
  {
    const auto __estimates = __hyperloglog_ns::__raw_estimate_data(__precision);
    const auto __n         = static_cast<::cuda::std::int32_t>(__hyperloglog_ns::__raw_estimate_data_size(__precision));
    ::cuda::std::int32_t __left  = 0;
    ::cuda::std::int32_t __right = __n - 1;

    while (__left <= __right)
    {
      const ::cuda::std::int32_t __mid = ::cuda::std::midpoint(__left, __right);

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

    // At this point, '__left' is the binary-search insertion point. Spark uses the insertion
    // point as the anchor index when the exact estimate is not present in the table.
    return __left;
  }

  static constexpr ::cuda::std::int32_t __k = 6; ///< Number of interpolation points to consider
  ::cuda::std::int32_t __precision; ///< HLL precision parameter
  ::cuda::std::int32_t __m; ///< Number of registers (2^precision)
};
} // namespace cuda::experimental::cuco::__hyperloglog_ns

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___HYPERLOGLOG_FINALIZER_CUH
