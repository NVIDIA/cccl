//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_ELEMENT_H
#define _CUDA_STD___TUPLE_TUPLE_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

// cuda::std::tuple_element

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element;

template <size_t _Ip, class _Tp>
using tuple_element_t _CCCL_NODEBUG_ALIAS = typename tuple_element<_Ip, _Tp>::type;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const volatile tuple_element_t<_Ip, _Tp>;
};

_CCCL_END_NAMESPACE_CUDA_STD

// std::tuple_element

_CCCL_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp>
struct tuple_element;

#if _CCCL_FREESTANDING()
template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = volatile typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG_ALIAS = const volatile typename tuple_element<_Ip, _Tp>::type;
};
#endif // _CCCL_FREESTANDING()

_CCCL_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_ELEMENT_H
