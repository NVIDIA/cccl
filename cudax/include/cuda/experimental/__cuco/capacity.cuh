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
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/__detail/prime.hpp>
#include <cuda/experimental/__cuco/__detail/types.cuh>
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
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Size next_valid_capacity(_Size __requested)
{
  constexpr auto __stride = static_cast<_Size>(_ProbingScheme::cg_size * _BucketSize);
  const auto __cycles     = ::cuda::ceil_div(::cuda::std::max(__requested, static_cast<_Size>(1)), __stride);
  if constexpr (::cuda::experimental::cuco::is_double_hashing_v<_ProbingScheme>)
  {
    if (__cycles > ::cuda::std::numeric_limits<_Size>::max() / __stride)
    {
      _CCCL_THROW(::std::logic_error, "Invalid input capacity");
    }
    return static_cast<_Size>(
      ::cuda::experimental::cuco::__detail::__next_prime(static_cast<::cuda::std::uint64_t>(__cycles)) * __stride);
  }
  else
  {
    return static_cast<_Size>((__cycles + static_cast<_Size>(__requested == 0)) * __stride);
  }
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
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Size next_valid_capacity(_Size __requested, double __load_factor)
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
  return ::cuda::experimental::cuco::next_valid_capacity<_ProbingScheme, _BucketSize>(static_cast<_Size>(__scaled));
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
  return ::cuda::experimental::cuco::next_valid_capacity<_ProbingScheme, _BucketSize>(__capacity) == __capacity;
}

template <class _ProbingScheme,
          int _BucketSize,
          ::cuda::std::size_t _Capacity = ::cuda::experimental::cuco::dynamic_extent>
class valid_capacity;

template <class _ProbingScheme, int _BucketSize, class _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr valid_capacity<_ProbingScheme, _BucketSize>
make_valid_capacity(_Size __requested);

//! @brief A validated open-addressing capacity.
//!
//! Carries a capacity that is guaranteed valid for `_ProbingScheme` and `_BucketSize`. A static
//! `_Capacity` encodes the valid slot count in the type, so equal-rounding requests share one type
//! and the modular reduction folds to a constant; a dynamic descriptor stores the valid slot count.
//! Values are minted only by `cuco::make_valid_capacity`.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Capacity Valid slot count, or `cuda::std::dynamic_extent` for a runtime-sized descriptor
template <class _ProbingScheme, int _BucketSize, ::cuda::std::size_t _Capacity>
class valid_capacity
{
public:
  using probing_scheme_type = _ProbingScheme; ///< Probing scheme type
  using size_type           = ::cuda::std::size_t; ///< Size type

  static constexpr int bucket_size = _BucketSize; ///< Number of slots per bucket
  static constexpr int cg_size     = _ProbingScheme::cg_size; ///< Cooperative-group size

  static_assert(_Capacity == ::cuda::experimental::cuco::dynamic_extent
                  || ::cuda::experimental::cuco::is_valid_capacity<_ProbingScheme, _BucketSize>(_Capacity),
                "Capacity must be a valid open-addressing capacity; obtain it via cuco::make_valid_capacity");

  //! @brief Default-constructs a static capacity descriptor (the value is encoded in the type).
  template <::cuda::std::size_t _C                                                          = _Capacity,
            ::cuda::std::enable_if_t<_C != ::cuda::experimental::cuco::dynamic_extent, int> = 0>
  _CCCL_HOST_DEVICE_API constexpr valid_capacity() noexcept
  {}

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type capacity() const noexcept
  {
    if constexpr (_Capacity != ::cuda::experimental::cuco::dynamic_extent)
    {
      return _Capacity;
    }
    else
    {
      return __capacity_;
    }
  }

  //! @brief Returns the number of buckets (`capacity() / bucket_size`).
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr size_type num_buckets() const noexcept
  {
    return capacity() / static_cast<size_type>(_BucketSize);
  }

  //! @brief Reduces `__x` modulo the capacity, the single reduction on the probing hot path.
  [[nodiscard]] friend _CCCL_HOST_DEVICE_API constexpr size_type
  operator%(size_type __x, const valid_capacity& __cap) noexcept
  {
    return __x % __cap.capacity();
  }

private:
  struct __empty_t
  {};
  using __storage_t =
    ::cuda::std::conditional_t<_Capacity == ::cuda::experimental::cuco::dynamic_extent, size_type, __empty_t>;
  _CCCL_NO_UNIQUE_ADDRESS __storage_t __capacity_{};

  //! @brief Private value constructor used by the runtime factory.
  template <::cuda::std::size_t _C                                                          = _Capacity,
            ::cuda::std::enable_if_t<_C == ::cuda::experimental::cuco::dynamic_extent, int> = 0>
  _CCCL_HOST_DEVICE_API explicit constexpr valid_capacity(size_type __capacity) noexcept
      : __capacity_{__capacity}
  {}

  template <class _PS, int _BS, class _S>
  friend _CCCL_HOST_DEVICE_API constexpr valid_capacity<_PS, _BS> make_valid_capacity(_S);
};

//! @brief Mints a dynamic validated capacity by rounding `__requested` up (runtime).
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Size Size type
//!
//! @param __requested Requested capacity
//!
//! @return A dynamic `valid_capacity<_ProbingScheme, _BucketSize>`
template <class _ProbingScheme, int _BucketSize, class _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr valid_capacity<_ProbingScheme, _BucketSize>
make_valid_capacity(_Size __requested)
{
  return valid_capacity<_ProbingScheme, _BucketSize>{
    static_cast<::cuda::std::size_t>(::cuda::experimental::cuco::next_valid_capacity<_ProbingScheme, _BucketSize>(
      static_cast<::cuda::std::size_t>(__requested)))};
}

//! @brief Mints a static validated capacity by rounding `_Requested` up (compile-time).
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Requested Requested capacity
//!
//! @return A static `valid_capacity<_ProbingScheme, _BucketSize, valid>`
template <class _ProbingScheme, int _BucketSize, ::cuda::std::size_t _Requested>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_valid_capacity()
{
  return valid_capacity<_ProbingScheme,
                        _BucketSize,
                        ::cuda::experimental::cuco::next_valid_capacity<_ProbingScheme, _BucketSize>(_Requested)>{};
}

//! @brief The static `valid_capacity` type for a probing scheme, bucket size, and requested capacity.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Number of slots per bucket
//! @tparam _Requested Requested capacity
template <class _ProbingScheme, int _BucketSize, ::cuda::std::size_t _Requested>
using valid_capacity_for_t =
  decltype(::cuda::experimental::cuco::make_valid_capacity<_ProbingScheme, _BucketSize, _Requested>());
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_CAPACITY_CUH
