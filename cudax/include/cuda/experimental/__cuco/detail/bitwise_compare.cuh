//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_BITWISE_COMPARE_CUH
#define _CUDAX___CUCO_DETAIL_BITWISE_COMPARE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_bitwise_comparable.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Bitwise equality comparison.
//!
//! @tparam _Tp Value type
template <class _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool __bitwise_compare(const _Tp& __lhs, const _Tp& __rhs)
{
  static_assert(::cuda::is_bitwise_comparable_v<_Tp>,
                "Bitwise compared objects must have unique object representations or be explicitly declared safe.");
  if constexpr (sizeof(_Tp) <= sizeof(::cuda::std::uint64_t)
                || (sizeof(_Tp) == 2 * sizeof(::cuda::std::uint64_t) && _CCCL_HAS_INT128()))
  {
    using _Up = ::cuda::std::__make_nbit_uint_t<sizeof(_Tp) * ::cuda::std::numeric_limits<unsigned char>::digits>;
    return ::cuda::std::bit_cast<_Up>(__lhs) == ::cuda::std::bit_cast<_Up>(__rhs);
  }
  else
  {
    using _Array = ::cuda::std::array<::cuda::std::uint64_t, sizeof(_Tp) / sizeof(::cuda::std::uint64_t)>;
    return ::cuda::std::bit_cast<_Array>(__lhs) == ::cuda::std::bit_cast<_Array>(__rhs);
  }
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_BITWISE_COMPARE_CUH
