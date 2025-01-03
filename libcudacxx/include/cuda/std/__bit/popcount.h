//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_POPCOUNT_H
#define _LIBCUDACXX___BIT_POPCOUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/popc.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<sizeof(_Tp) <= sizeof(uint64_t), int> __popcount(_Tp __t) noexcept
{
  using _Sp = _If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
  return _CUDA_VSTD::__cccl_popc(static_cast<_Sp>(__t));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<(sizeof(_Tp) > sizeof(uint64_t)), int> __popcount(_Tp __t) noexcept
{
  constexpr int _Ratio = sizeof(_Tp) / sizeof(uint64_t);
  int __count          = 0;
  for (int __i = 0; __i < _Ratio; ++__i)
  {
    __count += _CUDA_VSTD::__cccl_popc(static_cast<uint64_t>(__t));
    __t >>= numeric_limits<uint64_t>::digits;
  }
  return __count;
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, int>
popcount(_Tp __t) noexcept
{
  return _CUDA_VSTD::__popcount(__t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_POPCOUNT_H
