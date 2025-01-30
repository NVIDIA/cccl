//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_STORAGE_H
#define _CUDA___FLOATING_POINT_STORAGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/std/climits>
#  include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <size_t _NBits>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_make_storage_type()
{
  if constexpr (_NBits <= CHAR_BIT)
  {
    return _CUDA_VSTD::uint8_t{};
  }
  else if constexpr (_NBits <= 2 * CHAR_BIT)
  {
    return _CUDA_VSTD::uint16_t{};
  }
  else if constexpr (_NBits <= 4 * CHAR_BIT)
  {
    return _CUDA_VSTD::uint32_t{};
  }
  else if constexpr (_NBits <= 8 * CHAR_BIT)
  {
    return _CUDA_VSTD::uint64_t{};
  }
#  if !defined(_LIBCUDACXX_HAS_NO_INT128)
  else if constexpr (_NBits <= 16 * CHAR_BIT)
  {
    return ::__uint128_t{};
  }
#  endif // !_LIBCUDACXX_HAS_NO_INT128
  else
  {
    static_assert(__always_false<_NBits>(), "Unsupported number of bits for floating point type");
  }
}

template <size_t _NBits>
using __fp_storage_t = decltype(__fp_make_storage_type<_NBits>());

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_STORAGE_H
