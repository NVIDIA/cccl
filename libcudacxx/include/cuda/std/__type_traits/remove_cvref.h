//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_REMOVE_CVREF_H
#define _CUDA_STD___TYPE_TRAITS_REMOVE_CVREF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(remove_cvref) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX(14)
#  define _CCCL_BUILTIN_REMOVE_CVREF(...) __remove_cvref(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(remove_cvref) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX( 14)

#if _CCCL_COMPILER(NVRTC, <, 12, 4) // NVRTC below 12.4 fails to properly compile that builtin
#  undef _CCCL_BUILTIN_REMOVE_CVREF
#endif // _CCCL_COMPILER(NVRTC, <, 12, 4)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_REMOVE_CVREF) && !defined(_LIBCUDACXX_USE_REMOVE_CVREF_FALLBACK)

template <class _Tp>
struct remove_cvref
{
  using type _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_CVREF(_Tp);
};

#  if _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()
template <class _Tp>
using remove_cvref_t _CCCL_NODEBUG = typename remove_cvref<_Tp>::type;
#  else // ^^^ _CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() ^^^ / vvv !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS() vvv
template <class _Tp>
using remove_cvref_t _CCCL_NODEBUG = _CCCL_BUILTIN_REMOVE_CVREF(_Tp);
#  endif // !_CCCL_DISALLOW_BUILTIN_IN_TYPE_ALIAS()

#else // ^^^ _CCCL_BUILTIN_REMOVE_CVREF ^^^ / vvv !_CCCL_BUILTIN_REMOVE_CVREF vvv

template <class _Tp>
struct remove_cvref
{
  using type _CCCL_NODEBUG = remove_cv_t<remove_reference_t<_Tp>>;
};

template <class _Tp>
using remove_cvref_t _CCCL_NODEBUG = remove_cv_t<remove_reference_t<_Tp>>;

#endif // defined(_CCCL_BUILTIN_REMOVE_CVREF) && !defined(_LIBCUDACXX_USE_REMOVE_CVREF_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_REMOVE_CVREF_H
