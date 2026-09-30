//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_ALL_OF_H
#define _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_ALL_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-field-initializers")
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// provide a generic way to initialize an aggregate member
struct __any_aggregate_member
{
  template <typename _Tp>
  _CCCL_API constexpr operator _Tp&&() const;
};

// Computes the number of aggregate members
template <typename _Tp>
struct __aggregate_arity_impl
{
  // N == 0: skip _Up{} to avoid inconsistencies across compilers (deleted default constructor)
  template <typename _Self = __aggregate_arity_impl>
  _CCCL_API auto operator()() -> decltype(_Self{}(__any_aggregate_member{}));

  // N >= 1: SFINAE on _Up{_A0{}, _Args{}...}
  template <typename _A0,
            typename... _Args,
            typename _Up   = _Tp,
            typename       = decltype(_Up{_A0{}, (_Args{})...}),
            typename _Self = __aggregate_arity_impl>
  _CCCL_API auto operator()(_A0, _Args... __args) -> decltype(_Self{}(_A0{}, __args..., __any_aggregate_member{}));

  template <typename... _Args>
  _CCCL_API auto operator()(_Args...) const -> char (*)[sizeof...(_Args) + 1]; // return the number of members + 1
};

// Returns the number of aggregate members. Only meaningful when is_aggregate_v<_Tp> is true.
template <typename _Tp>
inline constexpr int __aggregate_arity_v = int{sizeof(*__aggregate_arity_impl<_Tp>{}())} - 2;

// Apply a Predicate to every aggregate member

// provide a generic way to initialize an aggregate member but only if the Predicate is true
template <template <typename> class _Predicate>
struct __aggregate_member_if
{
  template <typename _Tp, typename = enable_if_t<_Predicate<remove_cvref_t<_Tp>>::value>>
  _CCCL_API constexpr operator _Tp&&() const;
};

inline constexpr int __aggregate_max_arity = 8;

// Apply the Predicate to every member
template <int _Arity>
struct __aggregate_all_of_fn
{
  template <template <typename> class _Predicate, typename _Tp, size_t... _Is>
  _CCCL_API static auto __test(index_sequence<_Is...>, int)
    -> decltype(_Tp{((void) _Is, __aggregate_member_if<_Predicate>{})...}, true_type{});

  template <template <typename> class, typename>
  _CCCL_API static auto __test(...) -> false_type;

  template <template <typename> class _Predicate, typename _Tp>
  _CCCL_API static auto __call(int) -> decltype(__test<_Predicate, _Tp>(make_index_sequence<_Arity>{}, 0));
};

// (aggregate-only trait) Dispatch based on the aggregate arity for non-empty aggregates. Returns:
// - false: _Arity out of [1, __aggregate_max_arity]
// - true:  if the unary predicate is true for all members
template <template <typename> class _Predicate,
          typename _Tp,
          int _Arity = __aggregate_arity_v<_Tp>,
          bool       = (_Arity > 0) && (_Arity <= __aggregate_max_arity)>
inline constexpr bool __aggregate_all_of_dispatch_v = false;

template <template <typename> class _Predicate, typename _Tp, int _Arity>
inline constexpr bool __aggregate_all_of_dispatch_v<_Predicate, _Tp, _Arity, true> =
  decltype(__aggregate_all_of_fn<_Arity>::template __call<_Predicate, _Tp>(0))::value;

// if _Tp is not an aggregate, return false.
// Empty aggregates are true for any predicate.
// The specialization is needed to skip computing the arity and save compile time.
template <template <typename> class _Predicate,
          typename _Tp,
          bool = is_aggregate_v<_Tp>,
          bool = is_empty_v<_Tp>> // required for nvc++ and NVCC+clang
inline constexpr bool __aggregate_all_of_v = false;

// (non-empty) aggregate
template <template <typename> class _Predicate, typename _Tp>
inline constexpr bool __aggregate_all_of_v<_Predicate, _Tp, true, false> =
  __aggregate_all_of_dispatch_v<_Predicate, _Tp>;

// Empty aggregate (true for any predicate)
template <template <typename> class _Predicate, typename _Tp>
inline constexpr bool __aggregate_all_of_v<_Predicate, _Tp, true, true> = true;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_AGGREGATE_MEMBERS_ALL_OF_H
