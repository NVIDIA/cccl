//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_TYPES_H
#define _CUDA_STD___TUPLE_TUPLE_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Tp>
struct __tuple_types
{};

// specialize cuda::std::tuple_size and cuda::std::tuple_element for cuda::std::__tuple_types

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<__tuple_types<_Tp...>> : integral_constant<size_t, sizeof...(_Tp)>
{};

template <size_t _Ip, class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_element<_Ip, __tuple_types<_Types...>> : enable_if<(_Ip < sizeof...(_Types)), __type_index_c<_Ip, _Types...>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_TYPES_H
