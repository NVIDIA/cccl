//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_REMOVE_VOLATILE) && !defined(_LIBCUDACXX_USE_REMOVE_VOLATILE_FALLBACK)
template <class _Tp>
struct remove_volatile
{
  using type _LIBCUDACXX_NODEBUG_TYPE = _CCCL_BUILTIN_REMOVE_VOLATILE(_Tp);
};

template <class _Tp>
using __remove_volatile_t = _CCCL_BUILTIN_REMOVE_VOLATILE(_Tp);

#else
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_volatile
{
  typedef _Tp type;
};
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_volatile<volatile _Tp>
{
  typedef _Tp type;
};

template <class _Tp>
using __remove_volatile_t = typename remove_volatile<_Tp>::type;

#endif // defined(_CCCL_BUILTIN_REMOVE_VOLATILE) && !defined(_LIBCUDACXX_USE_REMOVE_VOLATILE_FALLBACK)

#if _CCCL_STD_VER > 2011
template <class _Tp>
using remove_volatile_t = __remove_volatile_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
