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
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>

#include <cuda/std/__cccl/prologue.h>

// cuda::std::tuple_element

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element;

template <size_t _Ip, class _Tp>
using tuple_element_t _CCCL_NODEBUG = typename tuple_element<_Ip, _Tp>::type;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG = const tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG = volatile tuple_element_t<_Ip, _Tp>;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG = const volatile tuple_element_t<_Ip, _Tp>;
};

// specialize cuda::std::tuple_element for tuple-like ::std:: types
#if _CCCL_HAS_HOST_STD_LIB()
template <size_t _Ip, class _Tp, size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::array<_Tp, _Np>>
{
  static_assert(_Ip < _Np, "Index out of bounds in cuda::std::tuple_element<> (std::array)");
  using type _CCCL_NODEBUG = _Tp;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::complex<_Tp>>
{
  static_assert(_Ip < 2, "Index out of bounds in cuda::std::tuple_element<std::complex<_Tp>>");
  using type _CCCL_NODEBUG = _Tp;
};

template <size_t _Ip, class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::pair<_Tp, _Up>>
{
  static_assert(_Ip < 2, "Index out of bounds in cuda::std::tuple_element<std::pair<_Tp, _Up>>");
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<0, ::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG = _Tp;
};
template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<1, ::std::pair<_Tp, _Up>>
{
  using type _CCCL_NODEBUG = _Up;
};

template <size_t _Ip, class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::tuple<_Tp...>>
{
  static_assert(_Ip < sizeof...(_Tp), "Index out of bounds in cuda::std::tuple_element<> (std::tuple)");
  using type _CCCL_NODEBUG = tuple_element_t<_Ip, tuple<_Tp...>>;
};
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_END_NAMESPACE_CUDA_STD

// std::tuple_element

_CCCL_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp>
struct tuple_element;

#if _CCCL_FREESTANDING()
template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const _Tp>
{
  using type _CCCL_NODEBUG = const typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, volatile _Tp>
{
  using type _CCCL_NODEBUG = volatile typename tuple_element<_Ip, _Tp>::type;
};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, const volatile _Tp>
{
  using type _CCCL_NODEBUG = const volatile typename tuple_element<_Ip, _Tp>::type;
};
#endif // _CCCL_FREESTANDING()

_CCCL_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_ELEMENT_H
