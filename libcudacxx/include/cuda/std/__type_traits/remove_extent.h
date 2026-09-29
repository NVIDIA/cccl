//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_EXTENT_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_EXTENT_H

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

#if _CCCL_CHECK_BUILTIN(remove_extent) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX(15)
#  define _CCCL_BUILTIN_REMOVE_EXTENT(...) __remove_extent(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(remove_extent) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX( 15)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_EXTENT) && !defined(_LIBCUDACXX_USE_REMOVE_EXTENT_FALLBACK)
template <class _Tp>
struct remove_extent
{
  using type _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_EXTENT(_Tp);
};

#  if _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()
template <class _Tp>
using remove_extent_t _CCCL_NODEBUG = typename remove_extent<_Tp>::type;
#  else // ^^^ _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() ^^^ / vvv !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() vvv
template <class _Tp>
using remove_extent_t _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_EXTENT(_Tp);
#  endif // !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()

#else
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_extent
{
  using type = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_extent<_Tp[]>
{
  using type = _Tp;
};
template <class _Tp, size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_extent<_Tp[_Np]>
{
  using type = _Tp;
};

template <class _Tp>
using remove_extent_t _CCCL_NODEBUG = typename remove_extent<_Tp>::type;

#endif // defined(_CCCL_BUILTIN_REMOVE_EXTENT) && !defined(_LIBCUDACXX_USE_REMOVE_EXTENT_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_EXTENT_H
