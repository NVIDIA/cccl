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

template <typename _SizeType, _SizeType _Start, _SizeType _Step, typename _Operator, _SizeType... _Indices, typename... _TArgs>
_CCCL_API constexpr void __static_for_impl(
  _Operator __op,
  _CUDA_VSTD::integer_sequence<_SizeType, _Indices...>,
  _TArgs&&... __args) noexcept(noexcept(__op(_CUDA_VSTD::integral_constant<_SizeType, 0>{}, __args...)))
{
  (__op(_CUDA_VSTD::integral_constant<_SizeType, (_Indices * _Step + _Start)>{}, __args...), ...);
}

template <typename _Tp, _Tp _Size, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(__op(_CUDA_VSTD::integral_constant<_Tp, 0>{}, __args...)))
{
  ::cuda::__static_for_impl<_Tp, 0, 1>(
    __op, _CUDA_VSTD::make_integer_sequence<_Tp, _Size>{}, _CUDA_VSTD::forward<_TArgs>(__args)...);
}

template <typename _Tp, _Tp _Start, _Tp _End, _Tp _Step = 1, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(__op(_CUDA_VSTD::integral_constant<_Tp, 0>{}, __args...)))
{
  using __seq_t = _CUDA_VSTD::make_integer_sequence<_Tp, (_End - _Start) / _Step>;
  ::cuda::__static_for_impl<_Tp, _Start, _Step>(__op, __seq_t{}, _CUDA_VSTD::forward<_TArgs>(__args)...);
}

template <auto _Size, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(__op(_CUDA_VSTD::integral_constant<decltype(_Size), decltype(_Size){0}>{}, __args...)))
{
  ::cuda::static_for<decltype(_Size), _Size>(__op, _CUDA_VSTD::forward<_TArgs>(__args)...);
}

template <auto _Start, auto _End, auto _Step = 1, typename _Operator, typename... _TArgs>
_CCCL_API constexpr void static_for(_Operator __op, _TArgs&&... __args) noexcept(
  noexcept(__op(_CUDA_VSTD::integral_constant<decltype(_Start), decltype(_Start){0}>{}, __args...)))
{
  using _Tp = decltype(_Start);
  ::cuda::static_for<_Tp, _Start, _Tp{_End}, _Tp{_Step}>(__op, _CUDA_VSTD::forward<_TArgs>(__args)...);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_STATIC_FOR_H
