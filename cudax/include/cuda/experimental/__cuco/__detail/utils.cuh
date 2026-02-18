//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_UTILS_CUH
#define _CUDAX___CUCO___DETAIL_UTILS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>

#include <cstring>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Counts least significant bits in a 32-bit value.
_CCCL_DEVICE inline ::cuda::std::int32_t
__count_least_significant_bits(::cuda::std::uint32_t __x, ::cuda::std::int32_t __n)
{
  return __popc(__x & ((1u << __n) - 1u));
}

//! @brief Converts pair to tuple.
template <class _Key, class _Value>
struct __slot_to_tuple
{
  template <class _Slot>
  _CCCL_DEVICE ::cuda::std::tuple<_Key, _Value> operator()(const _Slot& __slot)
  {
    return ::cuda::std::tuple<_Key, _Value>(__slot.first, __slot.second);
  }
};

//! @brief Device functor returning whether the input slot is filled.
//!
//! Template parameter:
//! - `_Key`: Key type

template <class _Key>
struct __slot_is_filled
{
  _Key __empty_key_sentinel;

  template <class _Slot>
  _CCCL_DEVICE bool operator()(const _Slot& __slot)
  {
    return !__detail::__bitwise_compare(::cuda::std::get<0>(__slot), __empty_key_sentinel);
  }
};

template <class _SizeType, class _HashType>
_CCCL_HOST_DEVICE constexpr _SizeType __to_positive(_HashType __hash)
{
  if constexpr (::cuda::std::is_signed_v<_SizeType>)
  {
    return ::cuda::std::abs(static_cast<_SizeType>(__hash));
  }
  else
  {
    return static_cast<_SizeType>(__hash);
  }
}

template <class _SizeType, class _HashType>
_CCCL_HOST_DEVICE constexpr _SizeType __sanitize_hash(_HashType __hash) noexcept
{
  if constexpr (::cuda::std::is_same_v<_HashType, ::cuda::std::array<::cuda::std::uint64_t, 2>>)
  {
    unsigned __int128 __ret{};
    std::memcpy(&__ret, &__hash, sizeof(unsigned __int128));
    return __to_positive<_SizeType>(static_cast<_SizeType>(__ret));
  }
  else
  {
    return __to_positive<_SizeType>(__hash);
  }
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_UTILS_CUH
