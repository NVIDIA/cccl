//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_SFINAE_HELPERS_H
#define _CUDA_STD___TUPLE_SFINAE_HELPERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <bool... _Preds>
struct __all_dummy;

template <bool... _Pred>
using __all = is_same<__all_dummy<_Pred...>, __all_dummy<((void) _Pred, true)...>>;

enum class __smf_availability
{
  __trivial,
  __available,
  __deleted,
};

// We need to synthesize the copy / move assignment if it would be implicitly deleted as a member of a class
// In that case _Tp would be copy assignable but _TestSynthesizeAssignment<_Tp> would not
// This happens e.g for reference types
template <class _Tp>
struct _TestSynthesizeAssignment
{
  _Tp __dummy;
};

template <class _Tp>
inline constexpr bool __must_synthesize_assignment_v =
  (is_copy_assignable_v<_Tp> && !is_copy_assignable_v<_TestSynthesizeAssignment<_Tp>>)
  || (is_move_assignable_v<_Tp> && !is_move_assignable_v<_TestSynthesizeAssignment<_Tp>>);

// We need to ensure that __tuple_impl_sfinae_helper is unique for every instantiation of __tuple_impl, so its templated
// on the impl
template <class _Impl, bool _AllCopyAssignable, bool _AllMoveAssignable>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper
{};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, false, true>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = delete;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = default;
};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, true, false>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = default;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = delete;
};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, false, false>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = delete;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = delete;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_SFINAE_HELPERS_H
