//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_PROBING_SCHEME_BASE_CUH
#define _CUDAX___CUCO_DETAIL_PROBING_SCHEME_BASE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Base class of public probing schemes.
//!
//! @tparam _CgSize Cooperative group size
template <int _CgSize>
class __probing_scheme_base
{
public:
  static constexpr int __cg_size = _CgSize;
};

//! @brief Maps a hash to an index in `[0, __size)` without a division.
//!
//! The hash is read as a fraction of its type's range and scaled by `__size` (multiply-high), so the
//! high bits of the hash pick the index. Narrower hashes are left-aligned to 64 bits; 128-bit hashes
//! contribute their high word.
//!
//! @tparam _HashType Unsigned hash type of at most 128 bits
//!
//! @param[in] __hash Hash value
//! @param[in] __size Exclusive upper bound
//! @return An index smaller than `__size`
template <class _HashType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t
__hash_to_index(_HashType __hash, ::cuda::std::size_t __size) noexcept
{
  static_assert(::cuda::std::is_unsigned_v<_HashType> && ::cuda::std::__num_bits_v<_HashType> <= 128,
                "The hash type must be an unsigned integer of at most 128 bits");
  ::cuda::std::uint64_t __fraction{};
  if constexpr (::cuda::std::__num_bits_v<_HashType> > 64)
  {
    __fraction = static_cast<::cuda::std::uint64_t>(__hash >> 64);
  }
  else
  {
    __fraction = ::cuda::std::uint64_t{__hash} << (64 - ::cuda::std::__num_bits_v<_HashType>);
  }
  return ::cuda::mul_hi<::cuda::std::uint64_t>(__fraction, __size);
}

//! @brief Maps a 128-bit array hash, low word first, to an index in `[0, __size)` using its high word.
//!
//! @param[in] __hash Hash value with the low word first
//! @param[in] __size Exclusive upper bound
//! @return An index smaller than `__size`
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t
__hash_to_index(const ::cuda::std::array<::cuda::std::uint64_t, 2>& __hash, ::cuda::std::size_t __size) noexcept
{
  return __hash_to_index(__hash[1], __size);
}

//! @brief Probing iterator class.
//!
//! Yields slot offsets and wraps modulo the total capacity (in slots). The capacity is held as a
//! `cuda::std::extents` so a static slot count folds the reduction to a constant.
//!
//! @tparam _CapacityExtent Capacity extent type (total slots), a `cuda::std::extents`
//! @tparam _StepExtent Probe-step extent type, a `cuda::std::extents` (static for linear probing)
template <class _CapacityExtent, class _StepExtent>
class __probing_iterator
{
public:
  using __capacity_extent_type = _CapacityExtent;
  using __step_extent_type     = _StepExtent;
  using __size_type            = typename _CapacityExtent::index_type;

  _CCCL_HOST_DEVICE_API constexpr __probing_iterator(
    __size_type __start, _StepExtent __step, _CapacityExtent __capacity) noexcept
      : __curr_index{__start}
      , __step_{__step}
      , __capacity_{__capacity}
  {}

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE_API constexpr auto operator*() const noexcept
  {
    return __curr_index;
  }

  _CCCL_DEVICE_API constexpr auto operator++() noexcept
  {
    // Probe construction guarantees index < capacity and step <= capacity, so advancing wraps at
    // most once. Subtract before adding to avoid overflowing either signed or unsigned index types.
    const auto __remaining = __capacity_.extent(0) - __curr_index;
    __curr_index =
      __step_.extent(0) >= __remaining ? __step_.extent(0) - __remaining : __curr_index + __step_.extent(0);
    return *this;
  }

  _CCCL_DEVICE_API constexpr auto operator++(int) noexcept
  {
    auto __temp = *this;
    ++(*this);
    return __temp;
  }
#endif // _CCCL_CUDA_COMPILATION()

private:
  __size_type __curr_index;
  _CCCL_NO_UNIQUE_ADDRESS _StepExtent __step_;
  _CCCL_NO_UNIQUE_ADDRESS _CapacityExtent __capacity_;
};
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_PROBING_SCHEME_BASE_CUH
