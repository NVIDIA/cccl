// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_EMPTY_H
#define _CUDA_STD___RANGES_EMPTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

// [range.prim.empty]

_CCCL_BEGIN_NAMESPACE_CPO(__empty)

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_empty = __workaround_52970<_Tp> && requires(_Tp&& __t) { bool(__t.empty()); };

template <class _Tp>
concept __can_invoke_size = !__member_empty<_Tp> && requires(_Tp&& __t) { ::cuda::std::ranges::size(__t); };

template <class _Tp>
concept __can_compare_begin_end = !__member_empty<_Tp> && !__can_invoke_size<_Tp> && requires(_Tp&& __t) {
  bool(::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t));
  { ::cuda::std::ranges::begin(__t) } -> forward_iterator;
};
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__member_empty_, requires(_Tp&& __t)(requires(__workaround_52970<_Tp>), (bool(__t.empty()))));

template <class _Tp>
_CCCL_CONCEPT __member_empty = _CCCL_FRAGMENT(__member_empty_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_invoke_size_,
                       requires(_Tp&& __t)(requires(!__member_empty<_Tp>), ((void) ::cuda::std::ranges::size(__t))));

template <class _Tp>
_CCCL_CONCEPT __can_invoke_size = _CCCL_FRAGMENT(__can_invoke_size_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __can_compare_begin_end_,
  requires(_Tp&& __t)(requires(!__member_empty<_Tp>),
                      requires(!__can_invoke_size<_Tp>),
                      (bool(::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t))),
                      requires(forward_iterator<decltype(::cuda::std::ranges::begin(__t))>)));

template <class _Tp>
_CCCL_CONCEPT __can_compare_begin_end = _CCCL_FRAGMENT(__can_compare_begin_end_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_empty<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t) const noexcept(noexcept(bool(__t.empty())))
  {
    return bool(__t.empty());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_invoke_size<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t) const noexcept(noexcept(::cuda::std::ranges::size(__t)))
  {
    return ::cuda::std::ranges::size(__t) == 0;
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_compare_begin_end<_Tp>)
  [[nodiscard]] _CCCL_API constexpr bool operator()(_Tp&& __t) const
    noexcept(noexcept(bool(::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t))))
  {
    return ::cuda::std::ranges::begin(__t) == ::cuda::std::ranges::end(__t);
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto empty = __empty::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_EMPTY_H
