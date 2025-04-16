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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/concepts.cuh>
#include <cuda/experimental/__async/sender/just_from.cuh>
#include <cuda/experimental/__async/sender/meta.cuh>
#include <cuda/experimental/__async/sender/rcvr_ref.cuh>
#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__async/sender/variant.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

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
struct _FUNCTION_MUST_RETURN_A_BOOLEAN_TESTABLE_VALUE;

struct __cond_t
{
  template <class _Pred, class _Then, class _Else>
  struct params
  {
    _Pred pred;
    _Then on_true;
    _Else on_false;
  };

private:
  template <class... _As>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto __mk_complete_fn(_As&&... __as) noexcept
  {
    return [&](auto __sink) noexcept {
      return __sink(static_cast<_As&&>(__as)...);
    };
  }

  template <class... _As>
  using __just_from_t = decltype(just_from(__cond_t::__mk_complete_fn(declval<_As>()...)));

  template <class _Pred, class _Then, class _Else, class... _Env>
  struct __either_sig_fn
  {
    template <class... _As>
    _CUDAX_API constexpr auto operator()() const
    {
      if constexpr (!_CUDA_VSTD::__is_callable_v<_Pred, _As&...>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, __cond_t),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Pred),
                                            _WITH_ARGUMENTS(_As & ...)>();
      }
      else if constexpr (!_CUDA_VSTD::is_convertible_v<_CUDA_VSTD::__call_result_t<_Pred, _As&...>, bool>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, __cond_t),
                                            _WHAT(_FUNCTION_MUST_RETURN_A_BOOLEAN_TESTABLE_VALUE),
                                            _WITH_FUNCTION(_Pred),
                                            _WITH_ARGUMENTS(_As & ...)>();
      }
      else
      {
        return concat_completion_signatures(
          get_completion_signatures<__call_result_t<_Then, __just_from_t<_As...>>, _Env...>(),
          get_completion_signatures<__call_result_t<_Else, __just_from_t<_As...>>, _Env...>());
      }
    }
  };

  template <class _Sndr, class _Rcvr, class _Pred, class _Then, class _Else>
  struct __opstate
  {
    using operation_state_concept = operation_state_t;
    using __params_t              = params<_Pred, _Then, _Else>;
    using __env_t                 = _FWD_ENV_T<env_of_t<_Rcvr>>;

    template <class... _As>
    using __opstate_t =        //
      _CUDA_VSTD::__type_list< //
        connect_result_t<__call_result_t<_Then, __just_from_t<_As...>>, __rcvr_ref<_Rcvr>>,
        connect_result_t<__call_result_t<_Else, __just_from_t<_As...>>, __rcvr_ref<_Rcvr>>>;

    using __next_ops_variant_t = //
      __value_types<completion_signatures_of_t<_Sndr, __env_t>, __opstate_t, __type_concat_into_quote<__variant>::__call>;

    _CUDAX_API __opstate(_Sndr&& __sndr, _Rcvr&& __rcvr, __params_t&& __params)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __params_{static_cast<__params_t&&>(__params)}
        , __op_{__async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_ref{*this})}
    {}

    _CUDAX_API void start() noexcept
    {
      __async::start(__op_);
    }

    template <class... _As>
    _CUDAX_API void set_value(_As&&... __as) noexcept
    {
      auto __just = just_from(__cond_t::__mk_complete_fn(static_cast<_As&&>(__as)...));
      _CUDAX_TRY( //
        ({        //
          if (static_cast<_Pred&&>(__params_.pred)(__as...))
          {
            auto& __op =
              __ops_.__emplace_from(connect, static_cast<_Then&&>(__params_.on_true)(__just), __rcvr_ref{__rcvr_});
            __async::start(__op);
          }
          else
          {
            auto& __op =
              __ops_.__emplace_from(connect, static_cast<_Else&&>(__params_.on_false)(__just), __rcvr_ref{__rcvr_});
            __async::start(__op);
          }
        }),
        _CUDAX_CATCH(...) //
        ({                //
          __async::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
        }) //
      )
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

    _CUDAX_API auto get_env() const noexcept -> __env_t
    {
      return get_env(__rcvr_);
    }

    _Rcvr __rcvr_;
    __params_t __params_;
    connect_result_t<_Sndr, __rcvr_ref<__opstate, __env_t>> __op_;
    __next_ops_variant_t __ops_;
  };

  template <class _Pred, class _Then, class _Else>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure;

public:
  template <class _Sndr, class _Pred, class _Then, class _Else>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

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
struct _CCCL_TYPE_VISIBILITY_DEFAULT __cond_t::__sndr_t
{
  using __params_t = __cond_t::params<_Pred, _Then, _Else>;
  _CCCL_NO_UNIQUE_ADDRESS __cond_t __tag_;
  __params_t __params_;
  _Sndr __sndr_;

  template <class _Self, class... _Env>
  _CUDAX_API static constexpr auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      return concat_completion_signatures(
        transform_completion_signatures(__child_completions, __either_sig_fn<_Pred, _Then, _Else, _Env...>{}),
        __eptr_completion());
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate<_Sndr, _Rcvr, _Pred, _Then, _Else>
  {
    return {static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), static_cast<__params_t&&>(__params_)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& -> __opstate<_Sndr const&, _Rcvr, _Pred, _Then, _Else>
  {
    return {__sndr_, static_cast<_Rcvr&&>(__rcvr), static_cast<__params_t&&>(__params_)};
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
  if constexpr (!dependent_sender<_Sndr>)
  {
    using __completions = completion_signatures_of_t<__sndr_t<_Sndr, _Pred, _Then, _Else>>;
    static_assert(__valid_completion_signatures<__completions>);
  }

  return __sndr_t<_Sndr, _Pred, _Then, _Else>{
    {},
    {static_cast<_Pred&&>(__pred), static_cast<_Then&&>(__then), static_cast<_Else&&>(__else)},
    static_cast<_Sndr&&>(__sndr)};
}

template <class _Pred, class _Then, class _Else>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __cond_t::__closure
{
  __cond_t::params<_Pred, _Then, _Else> __params_;

  template <class _Sndr>
  _CUDAX_TRIVIAL_API auto __mk_sender(_Sndr&& __sndr) //
    -> __sndr_t<_Sndr, _Pred, _Then, _Else>
  {
    if constexpr (!dependent_sender<_Sndr>)
    {
      using __completions = completion_signatures_of_t<__sndr_t<_Sndr, _Pred, _Then, _Else>>;
      static_assert(__valid_completion_signatures<__completions>);
    }

    return __sndr_t<_Sndr, _Pred, _Then, _Else>{
      {}, static_cast<__cond_t::params<_Pred, _Then, _Else>&&>(__params_), static_cast<_Sndr&&>(__sndr)};
  }

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
inline constexpr size_t structured_binding_size<__cond_t::__sndr_t<_Sndr, _Pred, _Then, _Else>> = 3;

using conditional_t = __cond_t;
_CCCL_GLOBAL_CONSTANT conditional_t conditional{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
