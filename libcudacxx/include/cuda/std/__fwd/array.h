//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_ARRAY_H
#define _CUDA_STD___FWD_ARRAY_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

#  if _CCCL_HOST_STD_LIB(STL)
template <class _Tp, size_t _Size>
class array;
#  elif _CCCL_HOST_STD_LIB(LIBSTDCXX, <, 11) && defined(_GLIBCXX_DEBUG)
// Before GCC 11, the debug mode of libstdc++ provides std::array through the inline namespace std::__debug
inline namespace __debug
{
template <class _Tp, size_t _Size>
struct array;
} // namespace __debug
#  else // ^^^ libstdc++ debug mode before GCC 11 ^^^ / vvv other host standard library vvv
template <class _Tp, size_t _Size>
struct array;
#  endif // ^^^ other host standard library ^^^

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

// ::cuda::std:: forward declaration

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, size_t _Size>
struct _CCCL_TYPE_VISIBILITY_DEFAULT array;

template <class _Tp>
inline constexpr bool __is_cuda_std_array_v = false;

template <class _Tp, size_t _Sz>
inline constexpr bool __is_cuda_std_array_v<array<_Tp, _Sz>> = true;

#if _CCCL_HAS_HOST_STD_LIB()
template <class _Tp>
inline constexpr bool __is_std_array_v = false;

template <class _Tp, size_t _Sz>
inline constexpr bool __is_std_array_v<::std::array<_Tp, _Sz>> = true;
#endif // _CCCL_HAS_HOST_STD_LIB()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_ARRAY_H
