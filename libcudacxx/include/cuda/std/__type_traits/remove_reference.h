//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_REFERENCE_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__remove_reference)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference(__VA_ARGS__)
#elif _CCCL_HAS_BUILTIN(__remove_reference_t) && _CCCL_CUDA_COMPILER(CLANG)
#  define _CCCL_BUILTIN_REMOVE_REFERENCE_T(...) __remove_reference_t(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__remove_reference_t)

#if _CCCL_COMPILER(NVRTC, <, 12, 4) // NVRTC below 12.4 fails to properly compile cuda::std::move with that
#  undef _CCCL_BUILTIN_REMOVE_REFERENCE_T
#endif // _CCCL_COMPILER(NVRTC, <, 12, 4)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_REFERENCE_T) && !defined(_LIBCUDACXX_USE_REMOVE_REFERENCE_T_FALLBACK)
template <class _Tp>
struct remove_reference
{
  using type _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_REFERENCE_T(_Tp);
};

#  if _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()
template <class _Tp>
using remove_reference_t _CCCL_NODEBUG = typename remove_reference<_Tp>::type;
#  else // ^^^ _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() ^^^ / vvv !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() vvv
template <class _Tp>
using remove_reference_t _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_REFERENCE_T(_Tp);
#  endif // !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()

#else // ^^^ _CCCL_BUILTIN_REMOVE_REFERENCE_T ^^^ / vvv !_CCCL_BUILTIN_REMOVE_REFERENCE_T vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference<_Tp&>
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference<_Tp&&>
{
  using type _CCCL_NODEBUG = _Tp;
};

template <class _Tp>
using remove_reference_t _CCCL_NODEBUG = typename remove_reference<_Tp>::type;

#endif // !_CCCL_BUILTIN_REMOVE_REFERENCE_T

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_REFERENCE_H
