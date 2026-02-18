//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_BITWISE_COMPARE_CUH
#define _CUDAX___CUCO___DETAIL_BITWISE_COMPARE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/traits.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
_CCCL_HOST_DEVICE inline int __cuda_memcmp(const void* __lhs, const void* __rhs, ::cuda::std::size_t __count)
{
  auto __lhs_c = reinterpret_cast<const unsigned char*>(__lhs);
  auto __rhs_c = reinterpret_cast<const unsigned char*>(__rhs);
  while (__count--)
  {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v < __rhs_v)
    {
      return -1;
    }
    if (__lhs_v > __rhs_v)
    {
      return 1;
    }
  }
  return 0;
}

template <::cuda::std::size_t _TypeSize>
struct __bitwise_compare_impl
{
  _CCCL_HOST_DEVICE static bool compare(const char* __lhs, const char* __rhs)
  {
    return __cuda_memcmp(__lhs, __rhs, _TypeSize) == 0;
  }
};

template <>
struct __bitwise_compare_impl<4>
{
  _CCCL_HOST_DEVICE static bool compare(const char* __lhs, const char* __rhs)
  {
    return *reinterpret_cast<const ::cuda::std::uint32_t*>(__lhs)
        == *reinterpret_cast<const ::cuda::std::uint32_t*>(__rhs);
  }
};

template <>
struct __bitwise_compare_impl<8>
{
  _CCCL_HOST_DEVICE static bool compare(const char* __lhs, const char* __rhs)
  {
    return *reinterpret_cast<const ::cuda::std::uint64_t*>(__lhs)
        == *reinterpret_cast<const ::cuda::std::uint64_t*>(__rhs);
  }
};

//! @brief Alignment helper for bitwise compare
//!
//! Template parameter:
//! - `_Tp`: Type to align

template <class _Tp>
_CCCL_HOST_DEVICE constexpr ::cuda::std::size_t __alignment() noexcept
{
  constexpr ::cuda::std::size_t __align = ::cuda::std::bit_ceil(sizeof(_Tp));
  return ::cuda::std::min(::cuda::std::size_t{16}, __align);
}

//! @brief Bitwise equality comparison
//!
//! Template parameter:
//! - `_Tp`: Value type

template <class _Tp>
_CCCL_HOST_DEVICE constexpr bool __bitwise_compare(_Tp __lhs, _Tp __rhs)
{
  static_assert(::cuda::experimental::cuco::is_bitwise_comparable_v<_Tp>,
                "Bitwise compared objects must have unique object representations or be explicitly declared safe.");

  alignas(__alignment<_Tp>()) _Tp __lhs_aligned{__lhs};
  alignas(__alignment<_Tp>()) _Tp __rhs_aligned{__rhs};
  return __bitwise_compare_impl<sizeof(_Tp)>::compare(
    reinterpret_cast<const char*>(&__lhs_aligned), reinterpret_cast<const char*>(&__rhs_aligned));
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_BITWISE_COMPARE_CUH
