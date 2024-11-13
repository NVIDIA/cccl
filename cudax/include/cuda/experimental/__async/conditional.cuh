//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_CONDITIONAL
#define __CUDAX_ASYNC_DETAIL_CONDITIONAL

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/just_from.cuh>
#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/variant.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

//! \file conditional.cuh
//! This file defines the \c conditional sender. \c conditional is a sender that
//! selects between two continuations based on the result of a predecessor. It
//! accepts a predecessor, a predicate, and two continuations. It passes the
//! result of the predecessor to the predicate. If the predicate returns \c true,
//! the result is passed to the first continuation; otherwise, it is passed to
//! the second continuation.
//!
//! By "continuation", we mean a so-called sender adaptor closure: a unary function
//! that takes a sender and returns a new sender. The expression `then(f)` is an
//! example of a continuation.

namespace cuda::experimental::__async
{
struct __cond_t
{
  template <class _Pred, class _Then, class _Else>
  struct __data
  {
    _Pred __pred_;
    _Then __then_;
    _Else __else_;
  };

  template <class... _Args>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto __mk_complete_fn(_Args&&... __args) noexcept
  {
    return [&](auto __sink) noexcept {
      return __sink(static_cast<_Args&&>(__args)...);
    };
  }

  template <class... _Args>
  using __just_from_t = decltype(just_from(__cond_t::__mk_complete_fn(__declval<_Args>()...)));

  template <class _Sndr, class _Rcvr, class _Pred, class _Then, class _Else>
  struct __opstate
  {
    using operation_state_concept = operation_state_t;

    _CUDAX_API friend env_of_t<_Rcvr> get_env(const __opstate* __self) noexcept
    {
      return get_env(__self->__rcvr_);
    }

    template <class... _Args>
    using __value_t = //
      transform_completion_signatures<
        completion_signatures_of_t<__call_result_t<_Then, __just_from_t<_Args...>>, __rcvr_ref_t<_Rcvr&>>,
        completion_signatures_of_t<__call_result_t<_Else, __just_from_t<_Args...>>, __rcvr_ref_t<_Rcvr&>>>;

    template <class... _Args>
    using __opstate_t = //
      _CUDA_VSTD::__type_list< //
        connect_result_t<__call_result_t<_Then, __just_from_t<_Args...>>, __rcvr_ref_t<_Rcvr&>>,
        connect_result_t<__call_result_t<_Else, __just_from_t<_Args...>>, __rcvr_ref_t<_Rcvr&>>>;

    using __next_ops_variant_t = //
      __value_types<completion_signatures_of_t<_Sndr, __opstate*>,
                    __opstate_t,
                    __type_concat_into_quote<__variant>::__call>;

    using completion_signatures = //
      transform_completion_signatures_of<_Sndr, __opstate*, __async::completion_signatures<>, __value_t>;

    _CUDAX_API __opstate(_Sndr&& __sndr, _Rcvr&& __rcvr, __data<_Pred, _Then, _Else>&& __data)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __data_{static_cast<__cond_t::__data<_Pred, _Then, _Else>>(__data)}
        , __op_{__async::connect(static_cast<_Sndr&&>(__sndr), this)}
    {}

    _CUDAX_API void start() noexcept
    {
      __async::start(__op_);
    }

    template <class... _Args>
    _CUDAX_API void set_value(_Args&&... __args) noexcept
    {
      if (static_cast<_Pred&&>(__data_.__pred_)(__args...))
      {
        auto& __op = __ops_.__emplace_from(
          connect,
          static_cast<_Then&&>(__data_.__then_)(just_from(__cond_t::__mk_complete_fn(static_cast<_Args&&>(__args)...))),
          __rcvr_ref(__rcvr_));
        __async::start(__op);
      }
      else
      {
        auto& __op = __ops_.__emplace_from(
          connect,
          static_cast<_Else&&>(__data_.__else_)(just_from(__cond_t::__mk_complete_fn(static_cast<_Args&&>(__args)...))),
          __rcvr_ref(__rcvr_));
        __async::start(__op);
      }
    }

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __error) noexcept
    {
      __async::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    _CUDAX_API void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    _Rcvr __rcvr_;
    __cond_t::__data<_Pred, _Then, _Else> __data_;
    connect_result_t<_Sndr, __opstate*> __op_;
    __next_ops_variant_t __ops_;
  };

  template <class _Sndr, class _Pred, class _Then, class _Else>
  struct __sndr_t;

  template <class _Pred, class _Then, class _Else>
  struct __closure
  {
    __cond_t::__data<_Pred, _Then, _Else> __data_;

    template <class _Sndr>
    _CUDAX_TRIVIAL_API auto __mk_sender(_Sndr&& __sndr) //
      -> __sndr_t<_Sndr, _Pred, _Then, _Else>;

    template <class _Sndr>
    _CUDAX_TRIVIAL_API auto operator()(_Sndr __sndr) //
      -> __sndr_t<_Sndr, _Pred, _Then, _Else>
    {
      return __mk_sender(static_cast<_Sndr&&>(__sndr));
    }

    template <class _Sndr>
    _CUDAX_TRIVIAL_API friend auto operator|(_Sndr __sndr, __closure&& __self) //
      -> __sndr_t<_Sndr, _Pred, _Then, _Else>
    {
      return __self.__mk_sender(static_cast<_Sndr&&>(__sndr));
    }
  };

  template <class _Sndr, class _Pred, class _Then, class _Else>
  _CUDAX_TRIVIAL_API auto operator()(_Sndr __sndr, _Pred __pred, _Then __then, _Else __else) const //
    -> __sndr_t<_Sndr, _Pred, _Then, _Else>;

  template <class _Pred, class _Then, class _Else>
  _CUDAX_TRIVIAL_API auto operator()(_Pred __pred, _Then __then, _Else __else) const
  {
    return __closure<_Pred, _Then, _Else>{
      {static_cast<_Pred&&>(__pred), static_cast<_Then&&>(__then), static_cast<_Else&&>(__else)}};
  }
};

template <class _Sndr, class _Pred, class _Then, class _Else>
struct __cond_t::__sndr_t
{
  __cond_t __tag_;
  __cond_t::__data<_Pred, _Then, _Else> __data_;
  _Sndr __sndr_;

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate<_Sndr, _Rcvr, _Pred, _Then, _Else>
  {
    return {static_cast<_Sndr&&>(__sndr_),
            static_cast<_Rcvr&&>(__rcvr),
            static_cast<__cond_t::__data<_Pred, _Then, _Else>&&>(__data_)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& -> __opstate<_Sndr const&, _Rcvr, _Pred, _Then, _Else>
  {
    return {__sndr_, static_cast<_Rcvr&&>(__rcvr), static_cast<__cond_t::__data<_Pred, _Then, _Else>&&>(__data_)};
  }

  _CUDAX_API env_of_t<_Sndr> get_env() const noexcept
  {
    return __async::get_env(__sndr_);
  }
};

template <class _Sndr, class _Pred, class _Then, class _Else>
_CUDAX_TRIVIAL_API auto __cond_t::operator()(_Sndr __sndr, _Pred __pred, _Then __then, _Else __else) const //
  -> __sndr_t<_Sndr, _Pred, _Then, _Else>
{
  if constexpr (__is_non_dependent_sender<_Sndr>)
  {
    using __completions = completion_signatures_of_t<__sndr_t<_Sndr, _Pred, _Then, _Else>>;
    static_assert(__is_completion_signatures<__completions>);
  }

  return __sndr_t<_Sndr, _Pred, _Then, _Else>{
    {},
    {static_cast<_Pred&&>(__pred), static_cast<_Then&&>(__then), static_cast<_Else&&>(__else)},
    static_cast<_Sndr&&>(__sndr)};
}

template <class _Pred, class _Then, class _Else>
template <class _Sndr>
_CUDAX_TRIVIAL_API auto __cond_t::__closure<_Pred, _Then, _Else>::__mk_sender(_Sndr&& __sndr) //
  -> __sndr_t<_Sndr, _Pred, _Then, _Else>
{
  if constexpr (__is_non_dependent_sender<_Sndr>)
  {
    using __completions = completion_signatures_of_t<__sndr_t<_Sndr, _Pred, _Then, _Else>>;
    static_assert(__is_completion_signatures<__completions>);
  }

  return __sndr_t<_Sndr, _Pred, _Then, _Else>{
    {}, static_cast<__cond_t::__data<_Pred, _Then, _Else>&&>(__data_), static_cast<_Sndr&&>(__sndr)};
}

_CCCL_GLOBAL_CONSTANT __cond_t conditional{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
