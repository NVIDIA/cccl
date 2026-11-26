//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_TUPLE_H
#define _CUDA_STD___FWD_TUPLE_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

// MSVC reports the MSVC STL's own explicit specialization std::tuple<> at the first declaration of std::tuple, which
// is this one unless <utility> was included before
#  if _CCCL_COMPILER(MSVC, >=, 19, 44)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(5285) // cannot declare a specialization for 'std::tuple'
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 44)

template <class...>
class tuple;

#  if _CCCL_COMPILER(MSVC, >=, 19, 44)
_CCCL_DIAG_POP
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 44)

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class...>
class _CCCL_TYPE_VISIBILITY_DEFAULT tuple;

template <class>
inline constexpr bool __is_tuple_of_iterator_references_v = false;

template <class>
inline constexpr bool __is_cuda_std_tuple = false;

template <class... _Types>
inline constexpr bool __is_cuda_std_tuple<tuple<_Types...>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_TUPLE_H
