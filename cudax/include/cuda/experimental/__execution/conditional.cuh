//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_CONDITIONAL
#define __CUDAX_EXECUTION_CONDITIONAL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/just_from.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

//! @file conditional.cuh
//! This file defines the @c conditional sender. @c conditional is a sender that
//! selects between two continuations based on the result of a predecessor. It
//! accepts a predecessor, a predicate, and two continuations. It passes the
//! result of the predecessor to the predicate. If the predicate returns @c true,
//! the result is passed to the first continuation; otherwise, it is passed to
//! the second continuation.
//!
//! By "continuation", we mean a so-called sender adaptor closure: a unary function
//! that takes a sender and returns a new sender. The expression `then(f)` is an
//! example of a continuation.

namespace cuda::experimental::execution
{
struct _FUNCTION_MUST_RETURN_A_BOOLEAN_TESTABLE_VALUE;

struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Pred, class _Then, class _Else>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_base_t;

  template <class... _As>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto __mk_complete_fn(_As&&... __as) noexcept
  {
    return [&](auto __sink) noexcept {
      return __sink(static_cast<_As&&>(__as)...);
    };
  }

  template <class... _As>
  using __just_from_t _CCCL_NODEBUG_ALIAS = decltype(just_from(conditional_t::__mk_complete_fn(declval<_As>()...)));

  template <class _Pred, class _Then, class _Else, class... _Env>
  struct __either_sig_fn
  {
    template <class... _As>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (!__callable<_Pred, _As&...>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, conditional_t),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Pred),
                                            _WITH_ARGUMENTS(_As & ...)>();
      }
      else if constexpr (!::cuda::std::is_convertible_v<__call_result_t<_Pred, _As&...>, bool>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, conditional_t),
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

  template <class _Rcvr, class _Pred, class _Then, class _Else, class _Completions>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    using __params_t = __closure_base_t<_Pred, _Then, _Else>;

    template <class... _As>
    using __opstate_list_t =
      ::cuda::std::__type_list<connect_result_t<__call_result_t<_Then, __just_from_t<_As...>>, __rcvr_ref_t<_Rcvr>>,
                               connect_result_t<__call_result_t<_Else, __just_from_t<_As...>>, __rcvr_ref_t<_Rcvr>>>;

    using __next_ops_variant_t _CCCL_NODEBUG_ALIAS =
      __value_types<_Completions, __opstate_list_t, __type_concat_into_quote<__variant>::__call>;

    _Rcvr __rcvr_;
    __params_t __params_;
    __next_ops_variant_t __ops_{};
  };

  template <class _Rcvr, class _Pred, class _Then, class _Else, class _Completions>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    _CCCL_EXEC_CHECK_DISABLE
    template <class... _As>
    _CCCL_API void set_value(_As&&... __as) noexcept
    {
      auto __just = just_from(conditional_t::__mk_complete_fn(static_cast<_As&&>(__as)...));
      _CCCL_TRY
      {
        if (static_cast<_Pred&&>(__state_->__params_.pred)(__as...))
        {
          auto& __op = __state_->__ops_.__emplace_from(
            connect, static_cast<_Then&&>(__state_->__params_.on_true)(__just), __ref_rcvr(__state_->__rcvr_));
          execution::start(__op);
        }
        else
        {
          auto& __op = __state_->__ops_.__emplace_from(
            connect, static_cast<_Else&&>(__state_->__params_.on_false)(__just), __ref_rcvr(__state_->__rcvr_));
          execution::start(__op);
        }
      }
      _CCCL_CATCH_ALL
      {
        execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), ::std::current_exception());
      }
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__error));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_t<_Rcvr, _Pred, _Then, _Else, _Completions>* __state_;
  };

  template <class _CvSndr, class _Rcvr, class _Pred, class _Then, class _Else>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __completions_t         = completion_signatures_of_t<_CvSndr, __fwd_env_t<env_of_t<_Rcvr>>>;
    using __params_t              = __closure_base_t<_Pred, _Then, _Else>;
    using __rcvr_t                = conditional_t::__rcvr_t<_Rcvr, _Pred, _Then, _Else, __completions_t>;

    _CCCL_API __opstate_t(_CvSndr&& __sndr, _Rcvr&& __rcvr, __params_t&& __params)
        : __state_{static_cast<_Rcvr&&>(__rcvr), static_cast<__params_t&&>(__params)}
        , __op_{execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t{&__state_})}
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__op_);
    }

    __state_t<_Rcvr, _Pred, _Then, _Else, __completions_t> __state_;
    connect_result_t<_CvSndr, __rcvr_t> __op_;
  };

public:
  template <class _Pred, class _Then, class _Else>
  using params _CCCL_NODEBUG_ALIAS = __closure_base_t<_Pred, _Then, _Else>;

  template <class _Params, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr, class _Pred, class _Then, class _Else>
  _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Pred __pred, _Then __then, _Else __else) const;

  template <class _Pred, class _Then, class _Else>
  _CCCL_NODEBUG_API constexpr auto operator()(_Pred __pred, _Then __then, _Else __else) const;
};

template <class _Pred, class _Then, class _Else, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional_t::__sndr_t<conditional_t::__closure_base_t<_Pred, _Then, _Else>, _Sndr>
{
  using __params_t _CCCL_NODEBUG_ALIAS = conditional_t::__closure_base_t<_Pred, _Then, _Else>;
  _CCCL_NO_UNIQUE_ADDRESS conditional_t __tag_;
  __params_t __params_;
  _Sndr __sndr_;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
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
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr, _Pred, _Then, _Else>
  {
    return {static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), static_cast<__params_t&&>(__params_)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto
  connect(_Rcvr __rcvr) const& -> __opstate_t<_Sndr const&, _Rcvr, _Pred, _Then, _Else>
  {
    return {__sndr_, static_cast<_Rcvr&&>(__rcvr), static_cast<__params_t&&>(__params_)};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }
};

template <class _Pred, class _Then, class _Else>
struct _CCCL_TYPE_VISIBILITY_DEFAULT conditional_t::__closure_base_t
{
  _Pred pred;
  _Then on_true;
  _Else on_false;

  template <class _Sndr>
  _CCCL_NODEBUG_API auto __mk_sender(_Sndr&& __sndr) -> __sndr_t<__closure_base_t, _Sndr>
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = __early_domain_of_t<_Sndr>;
    // If the incoming sender is non-dependent, we can check the completion signatures of
    // the composed sender immediately.
    if constexpr (!dependent_sender<_Sndr>)
    {
      __assert_valid_completion_signatures(get_completion_signatures<__sndr_t<__closure_base_t, _Sndr>>());
    }
    return transform_sender(
      __dom_t{},
      __sndr_t<__closure_base_t, _Sndr>{{}, static_cast<__closure_base_t&&>(*this), static_cast<_Sndr&&>(__sndr)});
  }

  template <class _Sndr>
  _CCCL_NODEBUG_API auto operator()(_Sndr __sndr) -> __sndr_t<__closure_base_t, _Sndr>
  {
    return __mk_sender(static_cast<_Sndr&&>(__sndr));
  }

  template <class _Sndr>
  _CCCL_NODEBUG_API friend auto operator|(_Sndr __sndr, __closure_base_t __self) -> __sndr_t<__closure_base_t, _Sndr>
  {
    return __self.__mk_sender(static_cast<_Sndr&&>(__sndr));
  }
};

template <class _Sndr, class _Pred, class _Then, class _Else>
_CCCL_NODEBUG_API constexpr auto conditional_t::operator()(_Sndr __sndr, _Pred __pred, _Then __then, _Else __else) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS    = __early_domain_of_t<_Sndr>;
  using __params_t _CCCL_NODEBUG_ALIAS = __closure_base_t<_Pred, _Then, _Else>;
  __params_t __params{static_cast<_Pred&&>(__pred), static_cast<_Then&&>(__then), static_cast<_Else&&>(__else)};
  return static_cast<__params_t&&>(__params).__mk_sender(static_cast<_Sndr&&>(__sndr));
}

template <class _Pred, class _Then, class _Else>
_CCCL_NODEBUG_API constexpr auto conditional_t::operator()(_Pred __pred, _Then __then, _Else __else) const
{
  return __closure_base_t<_Pred, _Then, _Else>{
    static_cast<_Pred&&>(__pred), static_cast<_Then&&>(__then), static_cast<_Else&&>(__else)};
}

template <class _Params, class _Sndr>
inline constexpr size_t structured_binding_size<conditional_t::__sndr_t<_Params, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT conditional_t conditional{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_CONDITIONAL
