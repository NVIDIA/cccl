//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_INTEGRAL_H
#define _LIBCUDACXX___BIT_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// #include <cuda/__ptx/instructions/bfind.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr uint32_t __bit_log2(_Tp __t) noexcept
{
  if (!is_constant_evaluated() && sizeof(_Tp) <= 8)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::ptx::bfind(__t);))
  }
  return numeric_limits<_Tp>::digits - 1 - _CUDA_VSTD::countl_zero(__t);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int bit_width(_Tp __t) noexcept
{
  // __bit_log2 returns 0xFFFFFFFF if __t == 0. Since unsigned overflow is well-defined, the result is -1 + 1 = 0
  using _Up  = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  auto __ret = _CUDA_VSTD::__bit_log2(static_cast<_Up>(__t)) + 1; // type of__ret is int
  _CCCL_BUILTIN_ASSUME((is_unsigned_v<_Tp> ? true : __ret >= 0) && __ret <= numeric_limits<_Tp>::digits);
  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp bit_ceil(_Tp __t) noexcept
{
  _CCCL_ASSERT(__t <= numeric_limits<_Tp>::max() / 2, "bit_ceil overflow");
  if (is_constant_evaluated() && __t <= 1)
  {
    return 1;
  }
  // if __t == 0, bit_width() applies to 0xFFFFFFFF and returns 32
  // In CUDA, unsigned{1} << 32 --> 0
  // The result is computed as max(1, bit_width(__t - 1)) because max() requires less instructions than ternary operator
  using _Up    = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  auto __width = _CUDA_VSTD::bit_width(static_cast<_Up>(__t - 1)); // type of __ret is _Up
  if (!is_constant_evaluated() && sizeof(_Tp) <= 8)
  {
    NV_IF_TARGET(NV_IS_DEVICE, //
                 (auto __ret = static_cast<_Tp>(_CUDA_VSTD::max(_Tp{1}, ::cuda::ptx::shl(_Tp{1}, __width))); //
                  _CCCL_BUILTIN_ASSUME(__ret >= __t && __ret <= __t * 2);
                  return __ret;))
  }
  auto __ret = static_cast<_Tp>(__t <= 1 ? 1 : _Up{1} << __width);
  _CCCL_BUILTIN_ASSUME(__ret >= __t && __ret <= __t * 2);
  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_unsigned_integer, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp bit_floor(_Tp __t) noexcept
{
  if (is_constant_evaluated() && __t == 0)
  {
    return 0;
  }
  using _Up   = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  auto __log2 = _CUDA_VSTD::__bit_log2(static_cast<_Up>(__t));
  // __bit_log2 returns 0xFFFFFFFF if __t == 0
  // (CUDA) shift returns 0 if the right operand is larger than the number of bits of the type
  // -> the result is 0 if __t == 0
  if (!is_constant_evaluated() && sizeof(_Tp) <= 8)
  {
    NV_IF_TARGET(NV_IS_DEVICE, //
                 (auto __ret = static_cast<_Tp>(::cuda::ptx::shl(_Tp{1}, __log2))); //
                 _CCCL_BUILTIN_ASSUME(__ret >= __t / 2 && __ret <= __t);
                 return __ret;)
  }
  auto __ret = __t == 0 ? 0 : _Tp{1} << __log2;
  _CCCL_BUILTIN_ASSUME(__ret >= __t / 2 && __ret <= __t);
  return __ret;
}

#undef _CCCL_BUILTIN_ASSUME

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_INTEGRAL_H
