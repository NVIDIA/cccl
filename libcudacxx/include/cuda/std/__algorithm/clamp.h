//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_CLAMP_H
#define _CUDA_STD___ALGORITHM_CLAMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// Some code bases define `clang`, `msvc` or `lifetimebound` as macros, which break _CCCL_LIFETIMEBOUND.
#ifdef clang
#  pragma push_macro("clang")
#  undef clang
#  define _LIBCUDACXX_POP_MACRO_clang
#endif // defined(clang)
#ifdef msvc
#  pragma push_macro("msvc")
#  undef msvc
#  define _LIBCUDACXX_POP_MACRO_msvc
#endif // defined(msvc)
#ifdef lifetimebound
#  pragma push_macro("lifetimebound")
#  undef lifetimebound
#  define _LIBCUDACXX_POP_MACRO_lifetimebound
#endif // defined(lifetimebound)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Compare>
[[nodiscard]] _CCCL_API constexpr const _Tp&
clamp(const _Tp& __v _CCCL_LIFETIMEBOUND,
      const _Tp& __lo _CCCL_LIFETIMEBOUND,
      const _Tp& __hi _CCCL_LIFETIMEBOUND,
      _Compare __comp)
{
  _CCCL_ASSERT(!__comp(__hi, __lo), "Bad bounds passed to cuda::std::clamp");
  return __comp(__v, __lo) ? __lo : __comp(__hi, __v) ? __hi : __v;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr const _Tp&
clamp(const _Tp& __v _CCCL_LIFETIMEBOUND, const _Tp& __lo _CCCL_LIFETIMEBOUND, const _Tp& __hi _CCCL_LIFETIMEBOUND)
{
  _CCCL_ASSERT(!(__hi < __lo), "Bad bounds passed to cuda::std::clamp");
  return __v < __lo ? __lo : __hi < __v ? __hi : __v;
}

_CCCL_END_NAMESPACE_CUDA_STD

#ifdef _LIBCUDACXX_POP_MACRO_lifetimebound
#  pragma pop_macro("lifetimebound")
#  undef _LIBCUDACXX_POP_MACRO_lifetimebound
#endif // defined(_LIBCUDACXX_POP_MACRO_lifetimebound)
#ifdef _LIBCUDACXX_POP_MACRO_msvc
#  pragma pop_macro("msvc")
#  undef _LIBCUDACXX_POP_MACRO_msvc
#endif // defined(_LIBCUDACXX_POP_MACRO_msvc)
#ifdef _LIBCUDACXX_POP_MACRO_clang
#  pragma pop_macro("clang")
#  undef _LIBCUDACXX_POP_MACRO_clang
#endif // defined(_LIBCUDACXX_POP_MACRO_clang)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_CLAMP_H
