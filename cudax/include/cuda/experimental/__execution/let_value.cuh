//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_LET_VALUE
#define __CUDAX_EXECUTION_LET_VALUE

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
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// Declare types to use for diagnostics:
struct _FUNCTION_MUST_RETURN_A_SENDER;

namespace __let
{
template <class _Sndr, class _Domain>
struct __attrs_t : __attrs_t<_Sndr, __nil>
{
  using __attrs_t<_Sndr, __nil>::query;

  [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept -> _Domain
  {
    return {};
  }
};

template <class _Sndr>
struct __attrs_t<_Sndr, __nil>
{
  template <class _Tag>
  _CCCL_API constexpr auto query(get_completion_scheduler_t<_Tag>) const = delete;

  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND(!_CUDA_VSTD::same_as<_Query, get_domain_t>)
                   _CCCL_AND __queryable_with<env_of_t<_Sndr>, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<env_of_t<_Sndr>, _Query>)
    -> __query_result_t<env_of_t<_Sndr>, _Query>
  {
    return execution::get_env(__sndr_).query(_Query{});
  }

  const _Sndr& __sndr_;
};
} // namespace __let

template <class _LetTag, class _SetTag>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __let_t
{
  _CUDAX_SEMI_PRIVATE :
  friend struct let_value_t;
  friend struct let_error_t;
  friend struct let_stopped_t;

  using __let_tag_t = _LetTag;
  using __set_tag_t = _SetTag;

  /// @brief Computes the type of a variant of tuples to hold the results of
  /// the predecessor sender.
  template <class _Completions, class _Env>
  using __results_t _CCCL_NODEBUG_ALIAS =
    __gather_completion_signatures<_Completions, __set_tag_t, _CUDA_VSTD::__decayed_tuple, __variant>;

  template <class _Fn>
  struct __sender2_fn
  {
    template <class... _As>
    using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<_Fn, _CUDA_VSTD::decay_t<_As>&...>;
  };

  template <class _Fn, class _Rcvr>
  struct __opstate_fn
  {
    template <class... _As>
    using __call _CCCL_NODEBUG_ALIAS =
      connect_result_t<_CUDA_VSTD::__call_result_t<_Fn, _CUDA_VSTD::decay_t<_As>&...>, __rcvr_ref_t<_Rcvr>>;
  };

  /// @brief Computes the type of a variant of operation states to hold
  /// the second operation state.
  template <class _Completions, class _Fn, class _Rcvr>
  using __opstate2_t _CCCL_NODEBUG_ALIAS =
    __gather_completion_signatures<_Completions, __set_tag_t, __opstate_fn<_Fn, _Rcvr>::template __call, __variant>;

  template <class _Rcvr, class _Fn, class _Completions>
  struct __state_t
  {
    _Rcvr __rcvr_;
    _Fn __fn_;
    __results_t<_Completions, __fwd_env_t<env_of_t<_Rcvr>>> __result_{};
    __opstate2_t<_Completions, _Fn, _Rcvr> __opstate2_{};
  };

  template <class _Rcvr, class _Fn, class _Completions>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class _Tag, class... _As>
    _CCCL_API void __complete(_Tag, _As&&... __as) noexcept
    {
      if constexpr (_Tag{} == __set_tag_t{})
      {
        _CCCL_TRY
        {
          // Store the results so the lvalue refs we pass to the function
          // will be valid for the duration of the async op.
          auto& __tupl =
            __state_->__result_.template __emplace<_CUDA_VSTD::__decayed_tuple<_As...>>(static_cast<_As&&>(__as)...);
          // Call the function with the results and connect the resulting
          // sender, storing the operation state in __state_->__opstate2_.
          auto& __next_op = __state_->__opstate2_.__emplace_from(
            execution::connect,
            _CUDA_VSTD::__apply(static_cast<_Fn&&>(__state_->__fn_), __tupl),
            __ref_rcvr(__state_->__rcvr_));
          execution::start(__next_op);
        }
        _CCCL_CATCH_ALL
        {
          execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), ::std::current_exception());
        }
      }
      else
      {
        // Forward the completion to the receiver unchanged.
        _Tag{}(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_As&&>(__as)...);
      }
    }

    template <class... _As>
    _CCCL_TRIVIAL_API void set_value(_As&&... __as) noexcept
    {
      __complete(execution::set_value, static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    _CCCL_TRIVIAL_API void set_error(_Error&& __error) noexcept
    {
      __complete(execution::set_error, static_cast<_Error&&>(__error));
    }

    _CCCL_TRIVIAL_API void set_stopped() noexcept
    {
      __complete(execution::set_stopped);
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr_));
    }

    __state_t<_Rcvr, _Fn, _Completions>* __state_;
  };

  /// @brief The `let_(value|error|stopped)` operation state.
  /// @tparam _CvSndr The cvref-qualified predecessor sender type.
  /// @tparam _Fn The function to be called when the predecessor sender
  /// completes.
  /// @tparam _Rcvr The receiver connected to the `let_(value|error|stopped)`
  /// sender.
  template <class _CvSndr, class _Fn, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __completions_t         = completion_signatures_of_t<_CvSndr, __fwd_env_t<env_of_t<_Rcvr>>>;
    using __rcvr_t                = __let_t::__rcvr_t<_Rcvr, _Fn, __completions_t>;

    _CCCL_API constexpr explicit __opstate_t(_CvSndr&& __sndr, _Fn __fn, _Rcvr __rcvr) noexcept(
      __nothrow_decay_copyable<_Fn, _Rcvr> && __nothrow_connectable<_CvSndr, __rcvr_t>)
        : __state_{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fn&&>(__fn)}
        , __opstate1_(execution::connect(static_cast<_CvSndr&&>(__sndr), __rcvr_t{&__state_}))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

    __state_t<_Rcvr, _Fn, __completions_t> __state_;
    connect_result_t<_CvSndr, __rcvr_t> __opstate1_;
  };

  template <class _Fn, class... _Env>
  struct __transform_args_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()() const
    {
      if constexpr (!__decay_copyable<_Ts...>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_ARGUMENTS_ARE_NOT_DECAY_COPYABLE),
                                            _WITH_ARGUMENTS(_Ts...)>();
      }
      else if constexpr (!_CUDA_VSTD::__is_callable_v<_Fn, _CUDA_VSTD::decay_t<_Ts>&...>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_FUNCTION_IS_NOT_CALLABLE),
                                            _WITH_FUNCTION(_Fn),
                                            _WITH_ARGUMENTS(_CUDA_VSTD::decay_t<_Ts> & ...)>();
      }
      else if constexpr (!__is_sender<_CUDA_VSTD::__call_result_t<_Fn, _CUDA_VSTD::decay_t<_Ts>&...>>)
      {
        return invalid_completion_signature<_WHERE(_IN_ALGORITHM, _LetTag),
                                            _WHAT(_FUNCTION_MUST_RETURN_A_SENDER),
                                            _WITH_FUNCTION(_Fn),
                                            _WITH_ARGUMENTS(_CUDA_VSTD::decay_t<_Ts> & ...)>();
      }
      else
      {
        // TODO: test that _Sndr satisfies sender_in<_Sndr, _Env...>
        using _Sndr _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<_Fn, _CUDA_VSTD::decay_t<_Ts>&...>;
        // The function is callable with the arguments and returns a sender, but we
        // do not know whether connect will throw.
        return concat_completion_signatures(get_completion_signatures<_Sndr, _Env...>(), __eptr_completion());
      }
    }
  };

  template <class _Fn>
  struct __domain_transform_fn
  {
    template <class... _Ts>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()(__set_tag_t (*)(_Ts...)) const
    {
      using __result_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<_Fn, _CUDA_VSTD::decay_t<_Ts>&...>;
      // ask the result sender if it knows where it will complete:
      return __detail::__domain_of_t<env_of_t<__result_t>, get_completion_scheduler_t<set_value_t>, __nil>{};
    }
  };

  struct __domain_reduce_fn
  {
    template <class... _Domains>
    [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()(_Domains...) const
    {
      using __fn_t = _CUDA_VSTD::__type_try_catch<_CUDA_VSTD::__type_quote<_CUDA_VSTD::common_type_t>,
                                                  _CUDA_VSTD::__type_always<__nil>>;
      return _CUDA_VSTD::__type_call<__fn_t, _Domains...>{};
    }
  };

  template <class _Sndr, class _Fn>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __get_completion_domain() noexcept
  {
    // we can know the completion domain for non-dependent senders
    using __completions = completion_signatures_of_t<_Sndr>;
    if constexpr (__valid_completion_signatures<__completions>)
    {
      return __completions{}.select(__set_tag_t{}).transform_reduce(__domain_transform_fn<_Fn>{}, __domain_reduce_fn{});
    }
    else
    {
      return __nil{};
    }
  }

  template <class _Sndr, class _Fn>
  using __completion_domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__get_completion_domain<_Sndr, _Fn>());

  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_base_t;

  template <class _Fn>
  struct _CCCL_VISIBILITY_HIDDEN __closure_base_t // hidden visibility because member __fn_ is hidden if it is an
                                                  // extended (host/device) lambda
  {
    template <class _Sndr>
    [[nodiscard]] _CCCL_TRIVIAL_API auto operator()(_Sndr __sndr) const
      -> _CUDA_VSTD::__call_result_t<__let_tag_t, _Sndr, _Fn>
    {
      return __let_tag_t{}(static_cast<_Sndr&&>(__sndr), __fn_);
    }

    template <class _Sndr>
    [[nodiscard]] _CCCL_TRIVIAL_API friend auto operator|(_Sndr __sndr, const __closure_base_t& __self)
      -> _CUDA_VSTD::__call_result_t<__let_tag_t, _Sndr, _Fn>
    {
      return __let_tag_t{}(static_cast<_Sndr&&>(__sndr), __self.__fn_);
    }

    _Fn __fn_;
  };

public:
  /// @brief The `let_(value|error|stopped)` sender.
  /// @tparam _Sndr The predecessor sender.
  /// @tparam _Fn The function to be called when the predecessor sender
  /// completes.
  template <class _Sndr, class _Fn>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr __sndr, _Fn __fn) const;

  template <class _Fn>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Fn __fn) const noexcept;
};

struct let_value_t : __let_t<let_value_t, set_value_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct let_error_t : __let_t<let_error_t, set_error_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

struct let_stopped_t : __let_t<let_stopped_t, set_stopped_t>
{
  template <class _Sndr, class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Fn>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;
};

template <class _LetTag, class _SetTag>
template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __let_t<_LetTag, _SetTag>::__sndr_base_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  using __domain_t                         = __completion_domain_of_t<_Sndr, _Fn>;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__child_completions) = get_child_completion_signatures<_Self, _Sndr, _Env...>())
    {
      if constexpr (__set_tag_t{} == execution::set_value)
      {
        return transform_completion_signatures(__child_completions, __transform_args_fn<_Fn, _Env...>{});
      }
      else if constexpr (__set_tag_t{} == execution::set_error)
      {
        return transform_completion_signatures(__child_completions, {}, __transform_args_fn<_Fn, _Env...>{});
      }
      else
      {
        return transform_completion_signatures(__child_completions, {}, {}, __transform_args_fn<_Fn, _Env...>{});
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CCCL_API auto
  connect(_Rcvr __rcvr) && noexcept(__nothrow_constructible<__opstate_t<_Sndr, _Fn, _Rcvr>, _Sndr, _Fn, _Rcvr>)
    -> __opstate_t<_Sndr, _Fn, _Rcvr>
  {
    return __opstate_t<_Sndr, _Fn, _Rcvr>(
      static_cast<_Sndr&&>(__sndr_), static_cast<_Fn&&>(__fn_), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& noexcept(
    __nothrow_constructible<__opstate_t<const _Sndr&, _Fn, _Rcvr>, const _Sndr&, const _Fn&, _Rcvr>)
    -> __opstate_t<const _Sndr&, _Fn, _Rcvr>
  {
    return __opstate_t<const _Sndr&, _Fn, _Rcvr>(__sndr_, __fn_, static_cast<_Rcvr&&>(__rcvr));
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __let::__attrs_t<_Sndr, __domain_t>
  {
    return {__sndr_};
  }

  _CCCL_NO_UNIQUE_ADDRESS __let_tag_t __tag_;
  _Fn __fn_;
  _Sndr __sndr_;
};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_value_t::__sndr_t
    : __let_t<let_value_t, set_value_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_error_t::__sndr_t
    : __let_t<let_error_t, set_error_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Sndr, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_stopped_t::__sndr_t
    : __let_t<let_stopped_t, set_stopped_t>::__sndr_base_t<_Sndr, _Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_value_t::__closure_t : __let_t<let_value_t, set_value_t>::__closure_base_t<_Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_error_t::__closure_t : __let_t<let_error_t, set_error_t>::__closure_base_t<_Fn>
{};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT let_stopped_t::__closure_t
    : __let_t<let_stopped_t, set_stopped_t>::__closure_base_t<_Fn>
{};

template <class... _Sndr>
using __all_non_dependent_t = _CUDA_VSTD::__fold_and<(!dependent_sender<_Sndr>) ...>;

template <class _LetTag, class _SetTag>
template <class _Sndr, class _Fn>
[[nodiscard]] _CCCL_TRIVIAL_API constexpr auto __let_t<_LetTag, _SetTag>::operator()(_Sndr __sndr, _Fn __fn) const
{
  using __sndr_t   = typename _LetTag::template __sndr_t<_Sndr, _Fn>;
  using __domain_t = __early_domain_of_t<_Sndr>;
  // If the incoming sender is non-dependent, we can check the completion signatures of
  // the composed sender immediately.
  if constexpr (!dependent_sender<_Sndr>)
  {
    // Although the input sender is not dependent, the sender(s) returned from the
    // function might be. Only do eager type-checking if all the possible senders returned
    // by the function are non-dependent. If any of them is dependent, we will defer the
    // type-checking to the point where the sender is connected.
    using __completions_t = completion_signatures_of_t<_Sndr>;
    constexpr bool __all_non_dependent =
      __gather_completion_signatures<__completions_t, _SetTag, __sender2_fn<_Fn>::template __call, __all_non_dependent_t>::
        value;

    if constexpr (__all_non_dependent)
    {
      __assert_valid_completion_signatures(get_completion_signatures<__sndr_t>());
    }
  }
  return transform_sender(__domain_t{}, __sndr_t{{{}, static_cast<_Fn&&>(__fn), static_cast<_Sndr&&>(__sndr)}});
}

template <class _LetTag, class _SetTag>
template <class _Fn>
[[nodiscard]] _CCCL_TRIVIAL_API constexpr auto __let_t<_LetTag, _SetTag>::operator()(_Fn __fn) const noexcept
{
  using __closure_t = typename _LetTag::template __closure_t<_Fn>;
  return __closure_t{{static_cast<_Fn&&>(__fn)}};
}

template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_value_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_error_t::__sndr_t<_Sndr, _Fn>> = 3;
template <class _Sndr, class _Fn>
inline constexpr size_t structured_binding_size<let_stopped_t::__sndr_t<_Sndr, _Fn>> = 3;

_CCCL_GLOBAL_CONSTANT auto let_value   = let_value_t{};
_CCCL_GLOBAL_CONSTANT auto let_error   = let_error_t{};
_CCCL_GLOBAL_CONSTANT auto let_stopped = let_stopped_t{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_LET_VALUE
