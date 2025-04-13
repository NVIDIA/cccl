// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_MOVE_H
#define _LIBCUDACXX___UTILITY_MOVE_H

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

#include <utility> // IWYU pragma: keep

#if (_CCCL_COMPILER(CLANG, >=, 15) || _CCCL_COMPILER(GCC, >=, 15) || _CCCL_COMPILER(MSVC, >=, 19, 36)) \
  && !defined(__CUDA_ARCH__)
#  define _CCCL_HAS_BUILTIN_STD_MOVE() 1
#else
#  define _CCCL_HAS_BUILTIN_STD_MOVE() 0
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _LIBCUDACXX_HIDE_FROM_ABI constexpr remove_reference_t<_Tp>&& move(_Tp&& __t) noexcept
{
  using _Up _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tp>;
  return static_cast<_Up&&>(__t);
}

#if _CCCL_HAS_BUILTIN_STD_MOVE()

// The compiler treats ::std::move[_if_noexcept] as a builtin function so it does not need
// to be instantiated and will be compiled away even at -O0.

// "using ::std::move" is commented out because this would also drag in the algorithm
// ::std::move(Iterator, Iterator, Iterator) which will conflict with cuda::std::move in
// <cuda/std/algorithm.h>.

/*using ::std::move;*/
using ::std::move_if_noexcept;

#else // ^^^ _CCCL_HAS_BUILTIN_STD_MOVE() ^^^ / vvv !_CCCL_HAS_BUILTIN_STD_MOVE() vvv

template <class _Tp>
using __move_if_noexcept_result_t =
  conditional_t<!is_nothrow_move_constructible<_Tp>::value && is_copy_constructible<_Tp>::value, const _Tp&, _Tp&&>;

template <class _Tp>
[[nodiscard]] _CCCL_INTRINSIC _LIBCUDACXX_HIDE_FROM_ABI constexpr __move_if_noexcept_result_t<_Tp>
move_if_noexcept(_Tp& __x) noexcept
{
  return _CUDA_VSTD::move(__x);
}

#endif // _CCCL_HAS_BUILTIN_STD_MOVE()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_MOVE_H
