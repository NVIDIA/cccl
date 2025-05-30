//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_STATIC_FOR_H
#define _LIBCUDACXX___UTILITY_STATIC_FOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _SizeType, typename _Operator, auto... _Indices, typename... _TArgs>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void
__static_for_impl(_Operator __op, _CUDA_VSTD::integer_sequence<_SizeType, _Indices...>, _TArgs&&... __args)
{
  (__op(_CUDA_VSTD::integral_constant<_SizeType, _Indices>{}, _CUDA_VSTD::forward<_TArgs>(__args)...), ...);
}

template <auto _Offset, auto _Step, typename _SizeType, auto... _Indices>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __offset_and_step(_CUDA_VSTD::integer_sequence<_SizeType, _Indices...>)
{
  return _CUDA_VSTD::integer_sequence<_SizeType, (_Indices * _Step + _Offset)...>{};
}

template <auto _Size, typename _Operator, typename... _TArgs>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void static_for(_Operator __op, _TArgs&&... __args)
{
  using __size_type = decltype(_Size);
  ::cuda::__static_for_impl<__size_type>(
    _CUDA_VSTD::forward<_Operator>(__op),
    _CUDA_VSTD::make_integer_sequence<__size_type, _Size>{},
    _CUDA_VSTD::forward<_TArgs>(__args)...);
}

template <auto _Start, decltype(_Start) _End, decltype(_Start) _Step = 1, typename _Operator, typename... _TArgs>
_LIBCUDACXX_HIDE_FROM_ABI constexpr void static_for(_Operator __op, _TArgs&&... __args)
{
  using __size_type    = decltype(_Start);
  using __seq_t        = _CUDA_VSTD::make_integer_sequence<__size_type, (_End - _Start) / _Step>;
  constexpr auto __seq = ::cuda::__offset_and_step<_Start, _Step>(__seq_t{});
  return ::cuda::__static_for_impl<__size_type>(
    _CUDA_VSTD::forward<_Operator>(__op), __seq, _CUDA_VSTD::forward<_TArgs>(__args)...);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_STATIC_FOR_H
