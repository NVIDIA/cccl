//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_REMOVE_REFERENCE_T) && !defined(_LIBCUDACXX_USE_REMOVE_REFERENCE_T_FALLBACK)
template <class _Tp>
struct remove_reference
{
  using type _LIBCUDACXX_NODEBUG_TYPE = _CCCL_BUILTIN_REMOVE_REFERENCE_T(_Tp);
};

#  if defined(_CCCL_COMPILER_GCC)
// error: use of built-in trait in function signature; use library traits instead
template <class _Tp>
using __libcpp_remove_reference_t = typename remove_reference<_Tp>::type;
#  else // ^^^ _CCCL_COMPILER_GCC ^^^^/  vvv !_CCCL_COMPILER_GCC
template <class _Tp>
using __libcpp_remove_reference_t = _CCCL_BUILTIN_REMOVE_REFERENCE_T(_Tp);
#  endif // !_CCCL_COMPILER_GCC

#else // ^^^ _CCCL_BUILTIN_REMOVE_REFERENCE_T ^^^ / vvv !_CCCL_BUILTIN_REMOVE_REFERENCE_T vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference<_Tp&>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_reference<_Tp&&>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;
};

template <class _Tp>
using __libcpp_remove_reference_t = typename remove_reference<_Tp>::type;

#endif // !_CCCL_BUILTIN_REMOVE_REFERENCE_T

#if _CCCL_STD_VER > 2011
template <class _Tp>
using remove_reference_t = __libcpp_remove_reference_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H
