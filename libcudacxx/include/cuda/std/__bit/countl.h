//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_COUNTL_H
#define _LIBCUDACXX___BIT_COUNTL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/clz.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp)
                 _CCCL_AND(sizeof(_Tp) >= sizeof(uint32_t) _CCCL_AND(sizeof(_Tp) <= sizeof(uint64_t))))
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<sizeof(_Tp) == sizeof(uint32_t) || sizeof(_Tp) == sizeof(uint64_t), int>
__countl_zero(_Tp __t) noexcept
{
  if (_CUDA_VSTD::is_constant_evaluated() && __t == 0)
  {
    return numeric_limits<_Tp>::digits;
  }
  using _Sp         = _If<sizeof(_Tp) == sizeof(uint32_t), uint32_t, uint64_t>;
  auto __clz_result = _CUDA_VSTD::__cccl_clz(static_cast<_Sp>(__t));
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (return __clz_result;), // if __t == 0 __clz_result is already equal to numeric_limits<_Tp>::digits
                    (return __t ? __clz_result : numeric_limits<_Tp>::digits;))
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp) _CCCL_AND(sizeof(_Tp) < sizeof(uint32_t)))
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_zero(_Tp __t) noexcept
{
  return _CUDA_VSTD::__countl_zero(static_cast<uint32_t>(__t))
       - (numeric_limits<uint32_t>::digits - numeric_limits<_Tp>::digits);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp) _CCCL_AND(sizeof(_Tp) > sizeof(uint64_t)))
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_zero(_Tp __t) noexcept
{
  constexpr int _Ratio = sizeof(_Tp) / sizeof(uint64_t);
  struct _Array
  {
    uint64_t __array[_Ratio];
  };
  auto __a = _CUDA_VSTD::bit_cast<_Array>(__t);
  for (int __i = _Ratio - 1; __i >= 0; --__i)
  {
    if (__a.__array[__i])
    {
      return _CUDA_VSTD::__countl_zero(__a.__array[__i]) + (_Ratio - 1 - __i) * numeric_limits<uint64_t>::digits;
    }
  }
  return numeric_limits<_Tp>::digits;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int countl_zero(_Tp __t) noexcept
{
  auto __ret = _CUDA_VSTD::__countl_zero(static_cast<_Tp>(__t));
  _CCCL_BUILTIN_ASSUME(__ret >= 0 && __ret <= numeric_limits<_Tp>::digits);
  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int countl_one(_Tp __t) noexcept
{
  auto __ret = _CUDA_VSTD::__countl_zero(static_cast<_Tp>(~__t));
  _CCCL_BUILTIN_ASSUME(__ret >= 0 && __ret <= numeric_limits<_Tp>::digits);
  return __ret;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_COUNTL_H
