//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_INHERIT_H
#define _CUDA___UTILITY_INHERIT_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// Some code bases define `empty_bases` as a macro, which would break _CCCL_DECLSPEC_EMPTY_BASES.
#ifdef empty_bases
#  pragma push_macro("empty_bases")
#  undef empty_bases
#  define _LIBCUDACXX_POP_MACRO_empty_bases
#endif // defined(empty_bases)

_CCCL_BEGIN_NAMESPACE_CUDA

template <class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_DECLSPEC_EMPTY_BASES __inherit : _Types...
{};

_CCCL_END_NAMESPACE_CUDA

#ifdef _LIBCUDACXX_POP_MACRO_empty_bases
#  pragma pop_macro("empty_bases")
#  undef _LIBCUDACXX_POP_MACRO_empty_bases
#endif // defined(_LIBCUDACXX_POP_MACRO_empty_bases)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_INHERIT_H
