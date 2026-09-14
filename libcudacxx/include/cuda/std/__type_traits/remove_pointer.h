//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_POINTER_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_POINTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(remove_pointer) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX(14)
#  define _CCCL_BUILTIN_REMOVE_POINTER(...) __remove_pointer(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(remove_pointer) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX( 14)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_POINTER) && !defined(_LIBCUDACXX_USE_REMOVE_POINTER_FALLBACK)
template <class _Tp>
struct remove_pointer
{
  using type _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_POINTER(_Tp);
};

#  if _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()
template <class _Tp>
using remove_pointer_t _CCCL_NODEBUG = typename remove_pointer<_Tp>::type;
#  else // ^^^ _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() ^^^ / vvv !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() vvv
template <class _Tp>
using remove_pointer_t _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_POINTER(_Tp);
#  endif // !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()

#else
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_pointer
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_pointer<_Tp*>
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_pointer<_Tp* const>
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_pointer<_Tp* volatile>
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_pointer<_Tp* const volatile>
{
  using type _CCCL_NODEBUG = _Tp;
};

template <class _Tp>
using remove_pointer_t _CCCL_NODEBUG = typename remove_pointer<_Tp>::type;

#endif // defined(_CCCL_BUILTIN_REMOVE_POINTER) && !defined(_LIBCUDACXX_USE_REMOVE_POINTER_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_POINTER_H
