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

#include <cuda/__cmath/uabs.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{
//! @brief Maps the unsigned magnitude of a hash to an index in `[0, __modulus)`.
//!
//! The unsigned magnitude is converted to the index width before computing the remainder. The
//! magnitude of a negative hash is computed in unsigned arithmetic, including the signed minimum.
//!
//! @param[in] __hash Hash value
//! @param[in] __modulus Positive exclusive upper bound
//! @return A nonnegative index smaller than `__modulus`
template <class _SizeType, class _HashType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType __hash_to_index(_HashType __hash, _SizeType __modulus) noexcept
{
  using __unsigned_size = ::cuda::std::make_unsigned_t<_SizeType>;
  return static_cast<_SizeType>(
    static_cast<__unsigned_size>(::cuda::uabs(__hash)) % static_cast<__unsigned_size>(__modulus));
}

//! @brief Maps the low 64-bit word of a 128-bit array hash to a valid index.
//!
//! @param[in] __hash Hash value with the low word first
//! @param[in] __modulus Positive exclusive upper bound
//! @return A nonnegative index smaller than `__modulus`
template <class _SizeType>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _SizeType
__hash_to_index(const ::cuda::std::array<::cuda::std::uint64_t, 2>& __hash, _SizeType __modulus) noexcept
{
  return ::cuda::experimental::cuco::detail::__hash_to_index(__hash[0], __modulus);
}
} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_UTILITY_HASH_TO_INDEX_CUH
