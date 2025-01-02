//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_COUNTR_H
#define _LIBCUDACXX___BIT_COUNTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/ctz.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<sizeof(_Tp) <= sizeof(uint64_t), int> __countr_zero(_Tp __t) noexcept
{
  using _Sp         = _If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
  auto __ctz_result = _CUDA_VSTD::__cccl_ctz(static_cast<_Sp>(__t));
  if (!__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __ctz_result;), (return __t ? __ctz_result : numeric_limits<_Tp>::digits;))
  }
  return __t ? __ctz_result : numeric_limits<_Tp>::digits;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<(sizeof(_Tp) > sizeof(uint64_t)), int> __countr_zero(_Tp __t) noexcept
{
  constexpr int _Ratio = sizeof(_Tp) / sizeof(uint64_t);
  struct _Array
  {
    uint64_t __array[_Ratio];
  };
  auto __a = _CUDA_VSTD::bit_cast<_Array>(__t);
  for (int __i = 0; __i < _Ratio; ++__i)
  {
    if (__a.__array[__i])
    {
      return _CUDA_VSTD::__countr_zero(__a.__array[__i]) + __i * numeric_limits<uint64_t>::digits;
    }
  }
  return numeric_limits<_Tp>::digits;
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, int>
countr_zero(_Tp __t) noexcept
{
  return _CUDA_VSTD::__countr_zero(__t);
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, int>
countr_one(_Tp __t) noexcept
{
  return _CUDA_VSTD::__countr_zero(static_cast<_Tp>(~__t));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_COUNTR_H
