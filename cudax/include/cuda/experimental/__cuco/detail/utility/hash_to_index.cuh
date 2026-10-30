//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_HASH_TO_INDEX_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_HASH_TO_INDEX_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Maps a hash to an index in `[0, __size)` without a division.
//!
//! The hash is interpreted as a fraction of the full range of its type and scaled by `__size`
//! (multiply-high), so the index is driven by the high bits of the hash.
//!
//! @tparam _HashType Unsigned hash type of at most 128 bits; a 128-bit hash contributes its high word
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
  constexpr int __hash_bits = ::cuda::std::__num_bits_v<_HashType>;
  // Narrower hashes are left-aligned and wider ones truncated to the high 64 bits, so that every hash
  // spans the full 64-bit fraction range.
  ::cuda::std::uint64_t __fraction{};
  if constexpr (__hash_bits > 64)
  {
    __fraction = static_cast<::cuda::std::uint64_t>(__hash >> 64);
  }
  else
  {
    __fraction = static_cast<::cuda::std::uint64_t>(__hash) << (64 - __hash_bits);
  }
  return static_cast<::cuda::std::size_t>(::cuda::mul_hi(__fraction, static_cast<::cuda::std::uint64_t>(__size)));
}

//! @brief Maps a 128-bit array hash to an index in `[0, __size)` using its high 64-bit word.
//!
//! @param[in] __hash Hash value with the low word first
//! @param[in] __size Exclusive upper bound
//! @return An index smaller than `__size`
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t
__hash_to_index(const ::cuda::std::array<::cuda::std::uint64_t, 2>& __hash, ::cuda::std::size_t __size) noexcept
{
  return ::cuda::experimental::cuco::detail::__hash_to_index(__hash[1], __size);
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_HASH_TO_INDEX_CUH
