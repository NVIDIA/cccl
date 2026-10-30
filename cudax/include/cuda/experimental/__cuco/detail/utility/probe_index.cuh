//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_UTILITY_PROBE_INDEX_CUH
#define _CUDAX___CUCO_DETAIL_UTILITY_PROBE_INDEX_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Maps the unsigned bit pattern of a hash to an index in `[0, __upper_bound)` using multiply-high.
//!
//! This is range reduction, not a remainder: the result is `floor(__hash * __upper_bound / 2^w)` for a `w`-bit
//! hash. Like a remainder, it gives every index either `floor(2^w / __upper_bound)` or `ceil(2^w / __upper_bound)`
//! hash values, but the high hash bits select the index, so the hash bits should be uniformly distributed. Hashes
//! wider than 64 bits are narrowed to the index width first.
//!
//! @param[in] __hash Hash value
//! @param[in] __upper_bound Positive exclusive upper bound
//! @return An index in `[0, __upper_bound)`
template <class _SizeType, class _HashType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType __probe_index(_HashType __hash, _SizeType __upper_bound) noexcept
{
  using __unsigned_size        = ::cuda::std::make_unsigned_t<_SizeType>;
  constexpr auto __hash_bytes  = sizeof(_HashType);
  constexpr auto __index_bytes = sizeof(_SizeType);
  if constexpr (__hash_bytes > sizeof(::cuda::std::uint64_t))
  {
    return ::cuda::experimental::cuco::detail::__probe_index(static_cast<__unsigned_size>(__hash), __upper_bound);
  }
  else if constexpr (__hash_bytes == sizeof(::cuda::std::uint32_t) && __index_bytes == sizeof(::cuda::std::uint64_t))
  {
    const auto __bits  = static_cast<::cuda::std::uint32_t>(__hash);
    const auto __bound = static_cast<::cuda::std::uint64_t>(__upper_bound);
    // Split the 96-bit product so that no 64-bit multiply-high is needed.
    return static_cast<_SizeType>(
      ::cuda::mul_hi(__bits, static_cast<::cuda::std::uint32_t>(__bound))
      + static_cast<::cuda::std::uint64_t>(__bits) * static_cast<::cuda::std::uint32_t>(__bound >> 32));
  }
  else if constexpr (__hash_bytes == sizeof(::cuda::std::uint64_t) && __index_bytes == sizeof(::cuda::std::uint32_t))
  {
    const auto __bits  = static_cast<::cuda::std::uint64_t>(__hash);
    const auto __bound = static_cast<::cuda::std::uint32_t>(__upper_bound);
    // Only the upper 32 bits of the 96-bit product are needed.
    const auto __middle = static_cast<::cuda::std::uint64_t>(static_cast<::cuda::std::uint32_t>(__bits >> 32)) * __bound
                        + ::cuda::mul_hi(static_cast<::cuda::std::uint32_t>(__bits), __bound);
    return static_cast<_SizeType>(__middle >> 32);
  }
  else
  {
    using __unsigned_hash = ::cuda::std::make_unsigned_t<_HashType>;
    using __product_type = ::cuda::std::conditional_t<(__hash_bytes > __index_bytes), __unsigned_hash, __unsigned_size>;
    constexpr auto __shift =
      ::cuda::std::numeric_limits<__product_type>::digits - ::cuda::std::numeric_limits<__unsigned_hash>::digits;
    // Left-align a narrower hash so that widening does not shrink the range of indices.
    const auto __bits =
      static_cast<__product_type>(static_cast<__product_type>(static_cast<__unsigned_hash>(__hash)) << __shift);
    return static_cast<_SizeType>(::cuda::mul_hi(__bits, static_cast<__product_type>(__upper_bound)));
  }
}

//! @brief Maps the low 64-bit word of a 128-bit array hash to an index in `[0, __upper_bound)`.
//!
//! @param[in] __hash Hash value with the low word first
//! @param[in] __upper_bound Positive exclusive upper bound
//! @return An index in `[0, __upper_bound)`
template <class _SizeType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType
__probe_index(const ::cuda::std::array<::cuda::std::uint64_t, 2>& __hash, _SizeType __upper_bound) noexcept
{
  using __unsigned_size = ::cuda::std::make_unsigned_t<_SizeType>;
  return ::cuda::experimental::cuco::detail::__probe_index(static_cast<__unsigned_size>(__hash[0]), __upper_bound);
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_PROBE_INDEX_CUH
