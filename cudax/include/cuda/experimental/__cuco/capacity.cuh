//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_CAPACITY_CUH
#define _CUDAX___CUCO_CAPACITY_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__numeric/mul_overflow.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/__detail/prime.hpp>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Trait value indicating whether a probing scheme is double hashing.
//!
//! @tparam _ProbingScheme Probing scheme type
template <class _ProbingScheme>
inline constexpr bool is_double_hashing_v = ::cuda::experimental::cuco::is_double_hashing<_ProbingScheme>::value;

//! @brief Rounds a requested capacity up to the smallest valid capacity for the given probing scheme
//! and bucket size.
//!
//! The probe stride is `_ProbingScheme::cg_size * _BucketSize`. For linear probing the result is a
//! multiple of the stride; for double hashing the probe cycle count `capacity / stride` is
//! additionally prime. The function is idempotent: applying it to an already valid capacity returns
//! the same value.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Size Size type
//!
//! @param __requested Requested capacity
//!
//! @return The smallest valid capacity that is greater than or equal to `__requested`
template <class _ProbingScheme, int _BucketSize, class _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Size make_valid_capacity(_Size __requested)
{
  constexpr auto __stride = static_cast<_Size>(_ProbingScheme::cg_size * _BucketSize);
  const auto __cycles     = ::cuda::ceil_div(::cuda::std::max(__requested, static_cast<_Size>(1)), __stride);
  _Size __capacity{};
  if constexpr (::cuda::experimental::cuco::is_double_hashing_v<_ProbingScheme>)
  {
    const auto __prime =
      ::cuda::experimental::cuco::__detail::__next_prime(static_cast<::cuda::std::uint64_t>(__cycles));
    if (::cuda::mul_overflow(__capacity, __prime, __stride))
    {
      _CCCL_THROW(::std::logic_error, "Invalid input capacity");
    }
  }
  else
  {
    const auto __num_buckets = __cycles + static_cast<_Size>(__requested == 0);
    if (::cuda::mul_overflow(__capacity, __num_buckets, __stride))
    {
      _CCCL_THROW(::std::logic_error, "Invalid input capacity");
    }
  }
  return __capacity;
}

//! @brief Rounds a requested capacity up to a valid capacity for a desired load factor.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Size Size type
//!
//! @param __requested Requested element count
//! @param __load_factor Desired load factor in (0, 1]
//!
//! @return The smallest valid capacity that fits `__requested` elements at `__load_factor`
template <class _ProbingScheme, int _BucketSize, class _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Size make_valid_capacity(_Size __requested, double __load_factor)
{
  if (__load_factor <= 0.)
  {
    _CCCL_THROW(::std::logic_error, "Desired load factor must be larger than zero");
  }
  if (__load_factor > 1.)
  {
    _CCCL_THROW(::std::logic_error, "Desired load factor must be no larger than one");
  }
  const auto __scaled = ::cuda::std::ceil(static_cast<double>(__requested) / __load_factor);
  if (__scaled > static_cast<double>(::cuda::std::numeric_limits<_Size>::max()))
  {
    _CCCL_THROW(::std::logic_error,
                "Invalid load factor: requested capacity divided by load factor exceeds the maximum representable "
                "value");
  }
  return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(static_cast<_Size>(__scaled));
}

//! @brief Returns whether `__capacity` is already a valid capacity for the given probing scheme and
//! bucket size.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Size Size type
//!
//! @param __capacity Capacity to test
//!
//! @return `true` if `__capacity` needs no rounding
template <class _ProbingScheme, int _BucketSize, class _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool is_valid_capacity(_Size __capacity)
{
  return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(__capacity) == __capacity;
}
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_CAPACITY_CUH
