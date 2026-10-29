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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__numeric/mul_overflow.h>
#include <cuda/__utility/in_range.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/detail/prime.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
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
//! @tparam _SizeType Size type
//!
//! @param[in] __requested Requested capacity
//!
//! @return The smallest valid capacity that is greater than or equal to `__requested`
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType make_valid_capacity(_SizeType __requested)
{
  static_assert(_ProbingScheme::cg_size > 0);
  static_assert(_BucketSize > 0);
  static_assert(::cuda::std::numeric_limits<_SizeType>::is_integer);
  static_assert(sizeof(_SizeType) <= sizeof(::cuda::std::uint64_t));

  constexpr auto __stride = static_cast<::cuda::std::uint64_t>(_ProbingScheme::cg_size) * _BucketSize;
  constexpr auto __max_cycles =
    static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<_SizeType>::max()) / __stride;
  const auto __requested_normalized = static_cast<::cuda::std::uint64_t>(::cuda::std::max(__requested, _SizeType{1}));
  const auto __cycles               = ::cuda::ceil_div(__requested_normalized, __stride);
  _SizeType __capacity{};
  if constexpr (is_double_hashing_v<_ProbingScheme>)
  {
    const auto __prime = ::cuda::experimental::cuco::detail::__next_prime(__cycles, __max_cycles);
    if (__prime == 0 || ::cuda::mul_overflow(__capacity, __prime, __stride))
    {
      _CCCL_THROW(::std::logic_error, "Invalid input capacity");
    }
  }
  else
  {
    const auto __num_buckets = __cycles + static_cast<::cuda::std::uint64_t>(__requested == 0);
    if (::cuda::mul_overflow(__capacity, __num_buckets, __stride))
    {
      _CCCL_THROW(::std::logic_error, "Invalid input capacity");
    }
  }
  return __capacity;
}

//! @brief Rounds a requested capacity up to a valid capacity for a desired load factor.
//!
//! For load factors less than one, scaling uses double-precision arithmetic. Rounding can affect
//! the resulting capacity for large requests. A scaled estimate outside the representable range
//! of `_SizeType` is rejected even when exact arithmetic would produce a representable capacity.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _SizeType Size type
//!
//! @param[in] __requested Requested element count
//! @param[in] __load_factor Desired load factor in (0, 1]
//!
//! @return The smallest valid capacity greater than or equal to the scaled estimate
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType make_valid_capacity(_SizeType __requested, double __load_factor)
{
  if (__load_factor <= 0. || !::cuda::in_range(__load_factor, 0., 1.))
  {
    _CCCL_THROW(::std::logic_error, "Desired load factor must be in the range (0, 1]");
  }
  if (__requested <= _SizeType{0} || __load_factor == 1.)
  {
    return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(__requested);
  }

  const auto __scaled = ::cuda::std::ceil(static_cast<double>(__requested) / __load_factor);
  // A 64-bit maximum rounds up to the next power of two when converted to double. Use an exactly
  // representable exclusive bound so the floating-to-integer conversion cannot overflow.
  constexpr auto __max             = ::cuda::std::numeric_limits<_SizeType>::max();
  constexpr auto __half_bound      = __max / 2 + 1;
  constexpr auto __exclusive_bound = static_cast<double>(__half_bound) * 2.;
  if (__scaled >= __exclusive_bound)
  {
    _CCCL_THROW(::std::logic_error,
                "Invalid load factor: requested capacity divided by load factor exceeds the maximum representable "
                "value");
  }
  return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(static_cast<_SizeType>(__scaled));
}

//! @brief Returns whether `__capacity` is already a valid capacity for the given probing scheme and
//! bucket size.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _SizeType Size type
//!
//! @param[in] __capacity Capacity to test
//!
//! @return `true` if `__capacity` needs no rounding
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool is_valid_capacity(_SizeType __capacity)
{
  return ::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize>(__capacity) == __capacity;
}
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_CAPACITY_CUH
