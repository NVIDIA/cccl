//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONCEPTS
#define __CUDAX_ASYNC_DETAIL_CONCEPTS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
// Utilities:
template <class _Ty>
_CUDAX_API constexpr auto __is_constexpr_helper(_Ty) -> bool
{
  return true;
}

// Receiver concepts:
template <class _Rcvr>
_CCCL_CONCEPT receiver =                                                              //
  _CCCL_REQUIRES_EXPR((_Rcvr))                                                        //
  (                                                                                   //
    requires(__is_receiver<_CUDA_VSTD::decay_t<_Rcvr>>),                              //
    requires(_CUDA_VSTD::move_constructible<_CUDA_VSTD::decay_t<_Rcvr>>),             //
    requires(_CUDA_VSTD::constructible_from<_CUDA_VSTD::decay_t<_Rcvr>, _Rcvr>),      //
    requires(_CUDA_VSTD::is_nothrow_move_constructible_v<_CUDA_VSTD::decay_t<_Rcvr>>) //
  );

template <class _Rcvr, class _Sig>
inline constexpr bool __valid_completion_for = false;

template <class _Rcvr, class _Tag, class... _As>
inline constexpr bool __valid_completion_for<_Rcvr, _Tag(_As...)> = _CUDA_VSTD::__is_callable_v<_Tag, _Rcvr, _As...>;

template <class _Rcvr, class _Completions>
inline constexpr bool __has_completions = false;

template <class _Rcvr, class... _Sigs>
inline constexpr bool __has_completions<_Rcvr, completion_signatures<_Sigs...>> =
  (__valid_completion_for<_Rcvr, _Sigs> && ...);

template <class _Rcvr, class _Completions>
_CCCL_CONCEPT receiver_of =                                               //
  _CCCL_REQUIRES_EXPR((_Rcvr, _Completions))                              //
  (                                                                       //
    requires(receiver<_Rcvr>),                                            //
    requires(__has_completions<_CUDA_VSTD::decay_t<_Rcvr>, _Completions>) //
  );

// Queryable traits:
template <class _Ty>
_CCCL_CONCEPT __queryable = _CUDA_VSTD::destructible<_Ty>;

// Awaitable traits:
template <class>
_CCCL_CONCEPT __is_awaitable = false; // TODO: Implement this concept.

// Sender traits:
template <class _Sndr>
_CUDAX_API constexpr auto __enable_sender() -> bool
{
  if constexpr (__is_sender<_Sndr>)
  {
    return true;
  }
  else
  {
    return __is_awaitable<_Sndr>;
  }
  _CCCL_UNREACHABLE();
}

template <class _Sndr>
inline constexpr bool enable_sender = __enable_sender<_Sndr>();

// Sender concepts:
template <class... _Env>
struct __completions_tester
{
  template <class _Sndr, bool EnableIfConstexpr = (get_completion_signatures<_Sndr, _Env...>(), true)>
  _CUDAX_API static constexpr auto __is_valid(int) -> bool
  {
    return __valid_completion_signatures<completion_signatures_of_t<_Sndr, _Env...>>;
  }

  template <class _Sndr>
  _CUDAX_API static constexpr auto __is_valid(long) -> bool
  {
    return false;
  }
};

template <class _Sndr>
_CCCL_CONCEPT sender =                                                          //
  _CCCL_REQUIRES_EXPR((_Sndr))                                                  //
  (                                                                             //
    requires(enable_sender<_CUDA_VSTD::decay_t<_Sndr>>),                        //
    requires(_CUDA_VSTD::move_constructible<_CUDA_VSTD::decay_t<_Sndr>>),       //
    requires(_CUDA_VSTD::constructible_from<_CUDA_VSTD::decay_t<_Sndr>, _Sndr>) //
  );

template <class _Sndr, class... _Env>
_CCCL_CONCEPT sender_in =                                                  //
  _CCCL_REQUIRES_EXPR((_Sndr, variadic _Env))                              //
  (                                                                        //
    requires(sender<_Sndr>),                                               //
    requires(sizeof...(_Env) <= 1),                                        //
    requires(_CUDA_VSTD::__fold_and_v<__queryable<_Env>...>),              //
    requires(__completions_tester<_Env...>::template __is_valid<_Sndr>(0)) //
  );

template <class _Sndr>
_CCCL_CONCEPT dependent_sender =             //
  _CCCL_REQUIRES_EXPR((_Sndr))               //
  (                                          //
    requires(sender<_Sndr>),                 //
    requires(__is_dependent_sender<_Sndr>()) //
  );

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_CONCEPTS
