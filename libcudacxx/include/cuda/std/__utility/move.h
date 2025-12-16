// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_MOVE_H
#define _CUDA_STD___UTILITY_MOVE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_reference.h>

#if _CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 12) || _CCCL_COMPILER(NVRTC)
#  define _CCCL_HAS_BUILTIN_STD_MOVE() 1
#else // ^^^ has builtin std::move ^^^ / vvv no builtin std::move vvv
#  define _CCCL_HAS_BUILTIN_STD_MOVE() 0
#endif // ^^^ no builtin std::move ^^^

// nvcc always supports std::move in device code.
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_MOVE
#  define _CCCL_HAS_BUILTIN_STD_MOVE() 1
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

#if _CCCL_COMPILER(CLANG, >=, 15)
#  define _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() 1
#else // ^^^ has builtin std::move_if_noexcept ^^^ / vvv no builtin std::move_if_noexcept vvv
#  define _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() 0
#endif // ^^^ no builtin std::move_if_noexcept ^^^

// nvcc warns about host only std::move_if_noexcept being used in device code
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT
#  define _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() 0
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

// include minimal std:: headers, nvcc in device mode doesn't need the std:: header
#if _CCCL_HAS_BUILTIN_STD_MOVE() || _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && _CCCL_HAS_INCLUDE(<bits/move.h>)
#    include <bits/move.h>
#  elif _CCCL_HOST_STD_LIB(LIBCXX) && _CCCL_HAS_INCLUDE(<__utility/move.h>)
#    include <__utility/move.h> // includes std::move_if_noexcept, too
#  elif !_CCCL_COMPILER(NVRTC)
#    include <utility>
#  endif
#endif // _CCCL_HAS_BUILTIN_STD_MOVE() || _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT()

// _CCCL_MOVE macro always expands to std::move builtin, if available, and fallbacks to cuda::std::move otherwise.
#if _CCCL_HAS_BUILTIN_STD_MOVE()
#  define _CCCL_MOVE(...) ::std::move(__VA_ARGS__)
#else // ^^^ _CCCL_HAS_BUILTIN_STD_MOVE() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_MOVE() vvv
#  define _CCCL_MOVE(...) ::cuda::std::move(__VA_ARGS__)
#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_MOVE() ^^^

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// We cannot simply use ::std::move here because it would drag in the move algorithm. This can work only if either/
// compiling with nvrtc or decorating the move algorithm with `requires true` which overrides the imported move
// algorithm. But this solution requires concepts.
#if _CCCL_HAS_BUILTIN_STD_MOVE() && (_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVRTC))

// Forward declare move algorithm decorated with `requires true` to override the imported move algorithm.
#  if _CCCL_HAS_CONCEPTS()
template <class _InputIterator, class _OutputIterator>
  requires true
_CCCL_API constexpr _OutputIterator move(_InputIterator __first, _InputIterator __last, _OutputIterator __result);
#  endif // _CCCL_HAS_CONCEPTS()

// The compiler treats ::std::move as a builtin function so it does not need to be instantiated and will be compiled
// away even at -O0.
using ::std::move;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_MOVE() && (_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVRTC)) ^^^ /
      // vvv !_CCCL_HAS_BUILTIN_STD_MOVE() || !(_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVRTC)) vvv

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr remove_reference_t<_Tp>&& move(_Tp&& __t) noexcept
{
  return static_cast<remove_reference_t<_Tp>&&>(__t);
}

#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_MOVE() || !(_CCCL_HAS_CONCEPTS() || _CCCL_COMPILER(NVRTC)) ^^^

#if _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT()

// The compiler treats ::std::move_if_noexcept as a builtin function so it does not need to be
// instantiated and will be compiled away even at -O0.
using ::std::move_if_noexcept;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() vvv

template <class _Tp>
using __move_if_noexcept_result_t =
  conditional_t<!is_nothrow_move_constructible_v<_Tp> && is_copy_constructible_v<_Tp>, const _Tp&, _Tp&&>;

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _CCCL_API constexpr __move_if_noexcept_result_t<_Tp> move_if_noexcept(_Tp& __x) noexcept
{
  return ::cuda::std::move(__x);
}

#endif // ^^^ !_CCCL_HAS_BUILTIN_STD_MOVE_IF_NOEXCEPT() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_MOVE_H
