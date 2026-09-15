//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H
#define _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// __make_tuple_types_t<_Tuple<_Types...>> is a __tuple_types<_Types...>

// MSVC eagerly substitutes an alias template that discards its index argument and then no longer
// sees a pack to expand.
#if _CCCL_COMPILER(MSVC)
template <class _Tp, size_t>
using __fake_type_at = type_identity_t<_Tp>;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
template <class _Tp, size_t>
using __fake_type_at = _Tp;
#endif // !_CCCL_COMPILER(MSVC)

template <class _Tp>
struct __make_tuple_types
{
  static_assert(__always_false_v<_Tp>, "Unsupported type in __make_tuple_types");
};

template <class>
struct __make_tuple_types_array;

template <size_t... _Indices>
struct __make_tuple_types_array<__tuple_indices<_Indices...>>
{
  template <class _Tp>
  using type _CCCL_NODEBUG = __tuple_types<__fake_type_at<_Tp, _Indices>...>;
};

template <class _Tp, size_t _Size>
struct __make_tuple_types<array<_Tp, _Size>>
{
  using type _CCCL_NODEBUG = typename __make_tuple_types_array<__make_tuple_indices_t<_Size>>::template type<_Tp>;
};

#if _CCCL_HAS_HOST_STD_LIB()
template <class _Tp, size_t _Size>
struct __make_tuple_types<::std::array<_Tp, _Size>>
{
  using type _CCCL_NODEBUG = typename __make_tuple_types_array<__make_tuple_indices_t<_Size>>::template type<_Tp>;
};
#endif _CCCL_HAS_HOST_STD_LIB()

template <class _Tp>
struct __make_tuple_types<complex<_Tp>>
{
  using type _CCCL_NODEBUG = __tuple_types<_Tp, _Tp>;
};

template <class _Tp>
struct __make_tuple_types<::cuda::complex<_Tp>>
{
  using type _CCCL_NODEBUG = __tuple_types<_Tp, _Tp>;
};

#if _CCCL_HAS_HOST_STD_LIB()
template <class _Tp>
struct __make_tuple_types<::std::complex<_Tp>>
{
  using type _CCCL_NODEBUG = __tuple_types<_Tp, _Tp>;
};
#endif _CCCL_HAS_HOST_STD_LIB()

template <template <class...> class _Tuple, class... _Types>
struct __make_tuple_types<_Tuple<_Types...>>
{
  using type _CCCL_NODEBUG = __tuple_types<_Types...>;
};

template <class _Tuple>
using __make_tuple_types_t = typename __make_tuple_types<remove_cvref_t<_Tuple>>::type;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H
