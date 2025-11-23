//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_CV_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_CV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_volatile.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(remove_cv)
#  define _CCCL_BUILTIN_REMOVE_CV(...) __remove_cv(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(remove_cv)

// gcc complains about __remove_cv in function signature
// clang < 20 + nvcc complains about __remove_cv being an undefined symbol
#if _CCCL_COMPILER(GCC) || (_CCCL_COMPILER(CLANG, <, 20) && _CCCL_CUDA_COMPILER(NVCC))
#  undef _CCCL_BUILTIN_REMOVE_CV
#endif // _CCCL_COMPILER(GCC) || (_CCCL_COMPILER(CLANG) && _CCCL_CUDA_COMPILER(NVCC))

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_CV)

template <class _Tp>
struct remove_cv
{
  using type _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_CV(_Tp);
};

template <class _Tp>
using remove_cv_t _CCCL_NODEBUG_ALIAS = _CCCL_BUILTIN_REMOVE_CV(_Tp);

#else // ^^^ _CCCL_BUILTIN_REMOVE_CV ^^^ / vvv !_CCCL_BUILTIN_REMOVE_CV vvv

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_cv
{
  using type = _Tp;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_cv<const _Tp>
{
  using type = _Tp;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_cv<volatile _Tp>
{
  using type = _Tp;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT remove_cv<const volatile _Tp>
{
  using type = _Tp;
};

template <class _Tp>
using remove_cv_t _CCCL_NODEBUG_ALIAS = remove_volatile_t<remove_const_t<_Tp>>;

#endif // ^^^ !_CCCL_BUILTIN_REMOVE_CV ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_CV_H
