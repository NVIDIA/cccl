//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_WHEN_ALL
#define __CUDAX_EXECUTION_WHEN_ALL

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
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/pod_tuple.h>
#include <cuda/std/atomic>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/lazy.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT when_all_t
{
  template <class... _Sndrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  _CUDAX_SEMI_PRIVATE :
  // Extract the first template parameter of the __state_t specialization.
  // The first template parameter is the receiver type.
  template <class _State>
  using __rcvr_from_state_t _CCCL_NODEBUG_ALIAS =
    ::cuda::std::__type_apply<::cuda::std::__detail::__type_at_fn<0>, _State>;

  // Returns the completion signatures of a child sender. Throws an exception if
  // the child sender has more than one set_value completion signature.
  template <class _Child, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __child_completions();

  // Merges the completion signatures of the child senders into a single set of
  // completion signatures for the when_all sender.
  template <class... _Completions>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __merge_completions(_Completions...);

  /// The receivers connected to the when_all's sub-operations expose this as
  /// their environment. Its `get_stop_token` query returns the token from
  /// when_all's stop source. All other queries are forwarded to the outer
  /// receiver's environment.
  template <class _StateZip>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_t
  {
    using __state_t _CCCL_NODEBUG_ALIAS = __unzip<_StateZip>;
    using __rcvr_t _CCCL_NODEBUG_ALIAS  = __rcvr_from_state_t<__state_t>;

    __state_t& __state_;

    [[nodiscard]] _CCCL_API constexpr auto query(get_stop_token_t) const noexcept -> inplace_stop_token
    {
      return __state_.__stop_token_;
    }

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query, class... _Args)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<__rcvr_t>, _Query, _Args...>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<__rcvr_t>, _Query, _Args...>)
        -> __query_result_t<env_of_t<__rcvr_t>, _Query, _Args...>
    {
      return execution::get_env(__state_.__rcvr_).query(_Query{}, static_cast<_Args&&>(__args)...);
    }
  };

  template <class _StateZip, size_t _Index>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept              = receiver_t;
    using __state_t _CCCL_NODEBUG_ALIAS = __unzip<_StateZip>;

    __state_t& __state_;

    template <class... _Ts>
    _CCCL_NODEBUG_API constexpr void set_value(_Ts&&... __ts) noexcept
    {
      constexpr ::cuda::std::index_sequence_for<_Ts...>* idx = nullptr;
      __state_.template __set_value<_Index>(idx, static_cast<_Ts&&>(__ts)...);
      __state_.__arrive();
    }

    template <class _Error>
    _CCCL_NODEBUG_API constexpr void set_error(_Error&& __error) noexcept
    {
      __state_.__set_error(static_cast<_Error&&>(__error));
      __state_.__arrive();
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      __state_.__set_stopped();
      __state_.__arrive();
    }

    _CCCL_API constexpr auto get_env() const noexcept -> __env_t<_StateZip>
    {
      return {__state_};
    }
  };

  enum __estate_t : int
  {
    __started,
    __error,
    __stopped
  };

  /// @brief The data stored in the operation state and referred to
  /// by the receiver.
  /// @tparam _Rcvr The receiver connected to the when_all sender.
  /// @tparam _CvFn A metafunction to apply cv- and ref-qualifiers to the senders
  /// @tparam _Sndrs A tuple of the when_all sender's child senders.
  template <class _Rcvr, class _CvFn, class _Sndrs>
  struct __state_t;

  template <class _Rcvr, class _CvFn, class _Ign0, class _Ign1, class... _Sndrs>
  struct __state_t<_Rcvr, _CvFn, ::cuda::std::__tuple<_Ign0, _Ign1, _Sndrs...>>
  {
    using __env_t _CCCL_NODEBUG_ALIAS     = when_all_t::__env_t<__zip<__state_t>>;
    using __sndr_t _CCCL_NODEBUG_ALIAS    = when_all_t::__sndr_t<_Sndrs...>;
    using __cv_sndr_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_call1<_CvFn, __sndr_t>;

    static constexpr auto __completions_and_offsets =
      __sndr_t::template __get_completions_and_offsets<__cv_sndr_t, __env_t>();

    using __completions_t _CCCL_NODEBUG_ALIAS = decltype(__completions_and_offsets.first);
    using __values_t _CCCL_NODEBUG_ALIAS = __value_types<__completions_t, __lazy_tuple, __type_self_or<__nil>::__call>;
    using __errors_t _CCCL_NODEBUG_ALIAS = __error_types<__completions_t, __variant>;
    using __stop_tok_t _CCCL_NODEBUG_ALIAS      = stop_token_of_t<env_of_t<_Rcvr>>;
    using __stop_callback_t _CCCL_NODEBUG_ALIAS = stop_callback_for_t<__stop_tok_t, __on_stop_request>;

    _CCCL_API explicit __state_t(_Rcvr __rcvr, size_t __count)
        : __rcvr_{static_cast<_Rcvr&&>(__rcvr)}
        , __count_{__count}
        , __stop_source_{}
        , __stop_token_{__stop_source_.get_token()}
        , __state_{__started}
        , __errors_{}
        , __values_{}
        , __on_stop_{}
    {}

    template <size_t _Index, size_t... _Jdx, class... _Ts>
    _CCCL_API void __set_value(::cuda::std::index_sequence<_Jdx...>*, [[maybe_unused]] _Ts&&... __ts) noexcept
    {
      if constexpr (!__same_as<__values_t, __nil>)
      {
        constexpr size_t _Offset = __completions_and_offsets.second[_Index];
        if constexpr (__nothrow_decay_copyable<_Ts...>)
        {
          (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
        }
        else
        {
          _CCCL_TRY
          {
            (__values_.template __emplace<_Jdx + _Offset>(static_cast<_Ts&&>(__ts)), ...);
          }
          _CCCL_CATCH_ALL
          {
            __set_error(::std::current_exception());
          }
        }
      }
    }

    template <class _Error>
    _CCCL_API void __set_error(_Error&& __err) noexcept
    {
      // TODO: Use weaker memory orders
      if (__error != __state_.exchange(__error))
      {
        __stop_source_.request_stop();
        // We won the race, free to write the error into the operation state
        // without worry.
        if constexpr (__nothrow_decay_copyable<_Error>)
        {
          __errors_.template __emplace<decay_t<_Error>>(static_cast<_Error&&>(__err));
        }
        else
        {
          _CCCL_TRY
          {
            __errors_.template __emplace<decay_t<_Error>>(static_cast<_Error&&>(__err));
          }
          _CCCL_CATCH_ALL
          {
            __errors_.template __emplace<::std::exception_ptr>(::std::current_exception());
          }
        }
      }
    }

    _CCCL_API void __set_stopped() noexcept
    {
      ::cuda::std::underlying_type_t<__estate_t> __expected = __started;
      // Transition to the "stopped" state if and only if we're in the
      // "started" state. (If this fails, it's because we're in an
      // error state, which trumps cancellation.)
      if (__state_.compare_exchange_strong(
            __expected, static_cast<::cuda::std::underlying_type_t<__estate_t>>(__stopped)))
      {
        __stop_source_.request_stop();
      }
    }

    _CCCL_API void __arrive() noexcept
    {
      if (0 == --__count_)
      {
        __complete();
      }
    }

    _CCCL_API void __complete() noexcept
    {
      // Stop callback is no longer needed. Destroy it.
      __on_stop_.destroy();
      // All child operations have completed and arrived at the barrier.
      switch (__state_.load(::cuda::std::memory_order_relaxed))
      {
        case __started:
          if constexpr (!__same_as<__values_t, __nil>)
          {
            // All child operations completed successfully:
            __values_.__apply(execution::set_value, static_cast<__values_t&&>(__values_), static_cast<_Rcvr&&>(__rcvr_));
          }
          break;
        case __error:
          // One or more child operations completed with an error:
          __errors_.__visit(execution::set_error, static_cast<__errors_t&&>(__errors_), static_cast<_Rcvr&&>(__rcvr_));
          break;
        case __stopped:
          execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
          break;
        default:;
      }
    }

    _Rcvr __rcvr_;
    ::cuda::std::atomic<size_t> __count_;
    inplace_stop_source __stop_source_;
    inplace_stop_token __stop_token_;
    ::cuda::std::atomic<::cuda::std::underlying_type_t<__estate_t>> __state_;
    __errors_t __errors_;
    __values_t __values_;
    __lazy<__stop_callback_t> __on_stop_;
  };

  struct __start_all
  {
    template <class... _Ops>
    _CCCL_NODEBUG_API void operator()(_Ops&... __ops) const noexcept
    {
      (execution::start(__ops), ...);
    }
  };

  /// The operation state for when_all
  template <class _Rcvr,
            class _CvFn,
            class _Sndrs,
            class = ::cuda::std::make_index_sequence<::cuda::std::__tuple_size_v<_Sndrs> - 2>>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t;

  template <class _Rcvr, class _CvFn, size_t... _Idx, class _Ign0, class _Ign1, class... _Sndrs>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT
  __opstate_t<_Rcvr, _CvFn, ::cuda::std::__tuple<_Ign0, _Ign1, _Sndrs...>, ::cuda::std::index_sequence<_Idx...>>
  {
    using operation_state_concept = operation_state_t;
    using __sndrs_t _CCCL_NODEBUG_ALIAS =
      ::cuda::std::__type_call<_CvFn, ::cuda::std::__tuple<_Ign0, _Ign1, _Sndrs...>>;
    using __state_t _CCCL_NODEBUG_ALIAS =
      when_all_t::__state_t<_Rcvr, _CvFn, ::cuda::std::__tuple<_Ign0, _Ign1, _Sndrs...>>;

    // This function object is used to connect all the sub-operations with
    // receivers, each of which knows which elements in the values tuple it
    // is responsible for setting.
    struct __connect_subs_fn
    {
      template <class... _CvSndrs>
      _CCCL_API constexpr auto
      operator()(__state_t& __state, ::cuda::std::__ignore_t, ::cuda::std::__ignore_t, _CvSndrs&&... __sndrs_) const
      {
        using __state_ref_t _CCCL_NODEBUG_ALIAS = __zip<__state_t>;
        // When there are no offsets, the when_all sender has no value
        // completions. All child senders can be connected to receivers
        // of the same type, saving template instantiations.
        [[maybe_unused]] constexpr bool __no_values =
          __same_as<decltype(__state_t::__completions_and_offsets.second), __nil>;
        // The offsets are used to determine which elements in the values
        // tuple each receiver is responsible for setting.
        return ::cuda::std::__tuple{execution::connect(
          static_cast<_CvSndrs&&>(__sndrs_), __rcvr_t<__state_ref_t, __no_values ? 0 : _Idx>{__state})...};
      }
    };

    // This is a tuple of operation states for the sub-operations.
    using __sub_opstates_t _CCCL_NODEBUG_ALIAS =
      ::cuda::std::__apply_result_t<__connect_subs_fn, __sndrs_t, __state_t&>;

    __state_t __state_;
    __sub_opstates_t __sub_ops_;

    /// Initialize the data member, connect all the sub-operations and
    /// save the resulting operation states in __sub_ops_.
    _CCCL_API constexpr explicit __opstate_t(__sndrs_t&& __sndrs_, _Rcvr __rcvr)
        : __state_{static_cast<_Rcvr&&>(__rcvr), sizeof...(_Sndrs)}
        , __sub_ops_{::cuda::std::__apply(__connect_subs_fn(), static_cast<__sndrs_t&&>(__sndrs_), __state_)}
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    /// Start all the sub-operations.
    _CCCL_API constexpr void start() noexcept
    {
      // register stop callback:
      __state_.__on_stop_.construct(
        get_stop_token(execution::get_env(__state_.__rcvr_)), __on_stop_request{__state_.__stop_source_});

      if (__state_.__stop_source_.stop_requested())
      {
        // Manually clean up the stop callback. We won't be starting the
        // sub-operations, so they won't complete and clean up for us.
        __state_.__on_stop_.destroy();

        // Stop has already been requested. Don't bother starting the child
        // operations.
        execution::set_stopped(static_cast<_Rcvr&&>(__state_.__rcvr_));
      }
      else
      {
        // Start all the sub-operations.
        ::cuda::std::__apply(__start_all{}, __sub_ops_);

        // If there are no sub-operations, we're done.
        if constexpr (sizeof...(_Sndrs) == 0)
        {
          __state_.__complete();
        }
      }
    }
  };

  template <class... _Ts>
  using __decay_all _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_list<decay_t<_Ts>...>;

public:
  template <class... _Sndrs>
  _CCCL_NODEBUG_API constexpr auto operator()(_Sndrs... __sndrs) const;
};

template <class _Child, class... _Env>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto when_all_t::__child_completions()
{
  using __env_t _CCCL_NODEBUG_ALIAS = prop<get_stop_token_t, inplace_stop_token>;
  _CUDAX_LET_COMPLETIONS(auto(__completions) = get_completion_signatures<_Child, env<__env_t, __fwd_env_t<_Env>>...>())
  {
    if constexpr (__completions.count(set_value) > 1)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, when_all_t),
                                          _WHAT(_SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS),
                                          _WITH_SENDER(_Child)>();
    }
    else
    {
      return __completions;
    }
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-value")

template <class... _Completions>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto when_all_t::__merge_completions(_Completions... __cs)
{
  // Use _CUDAX_LET_COMPLETIONS to ensure all completions are valid:
  _CUDAX_LET_COMPLETIONS(auto(__tmp) = (completion_signatures{}, ..., __cs)) // NB: uses overloaded comma operator
  {
    ::cuda::std::ignore          = __tmp; // silence unused variable warning
    auto __non_value_completions = concat_completion_signatures(
      completion_signatures<set_stopped_t()>{},
      transform_completion_signatures(__cs, __swallow_transform{}, __decay_transform<set_error_t>{})...);

    if constexpr (((0 == __cs.count(set_value)) || ...))
    {
      // at least one child sender has no value completions at all, so the
      // when_all will never complete with set_value. return just the error and
      // stopped completions.
      return ::cuda::std::__pair{__non_value_completions, __nil{}};
    }
    else
    {
      std::array<size_t, sizeof...(_Completions)> __offsets = {
        __value_types<_Completions, ::cuda::std::__type_list, ::cuda::std::__type_list_size>::value...};
      (void) ::cuda::std::exclusive_scan(__offsets.begin(), __offsets.end(), __offsets.begin(), std::size_t(0));

      // All child senders have exactly one value completion signature, each of
      // which may have multiple arguments. Concatenate all the arguments into a
      // single set_value_t completion signature.
      using __values_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_call< //
        __type_concat_into<__type_function<set_value_t>>, //
        __value_types<_Completions, __decay_all, ::cuda::std::__type_self_t>...>;
      // Add the value completion to the error and stopped completions.
      auto __local = __non_value_completions + completion_signatures<__values_t>();
      // Check if any of the values or errors are not nothrow decay-copyable.
      constexpr bool __all_nothrow_decay_copyable =
        (__value_types<_Completions, __nothrow_decay_copyable_t, ::cuda::std::type_identity_t>::value && ...);
      return ::cuda::std::__pair{__local + __eptr_completion_if<!__all_nothrow_decay_copyable>(), __offsets};
    }
  }

  _CCCL_UNREACHABLE();
}

_CCCL_DIAG_POP

// The sender for when_all
template <class... _Sndrs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT when_all_t::__sndr_t
    : ::cuda::std::__tuple<when_all_t, ::cuda::std::__ignore_t, _Sndrs...>
{
  using sender_concept                = sender_t;
  using __sndrs_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__tuple<when_all_t, ::cuda::std::__ignore_t, _Sndrs...>;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __get_completions_and_offsets()
  {
    return __merge_completions(__child_completions<::cuda::std::__copy_cvref_t<_Self, _Sndrs>, _Env...>()...);
  }

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return __get_completions_and_offsets<_Self, _Env...>().first;
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, __cp, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cp, __sndrs_t>(static_cast<__sndrs_t&&>(*this), static_cast<_Rcvr&&>(__rcvr));
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, __cpclr, __sndrs_t>
  {
    return __opstate_t<_Rcvr, __cpclr, __sndrs_t>(static_cast<__sndrs_t const&>(*this), static_cast<_Rcvr&&>(__rcvr));
  }

  struct _CCCL_TYPE_VISIBILITY_DEFAULT __attrs_t
  {
    [[nodiscard]] _CCCL_API constexpr auto query(get_domain_t) const noexcept
    {
      if constexpr (sizeof...(_Sndrs) == 0)
      {
        return default_domain{};
      }
      else
      {
        return ::cuda::std::common_type_t<__early_domain_of_t<_Sndrs>...>{};
      }
    }

    template <class... _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept
    {
      return (execution::min) (execution::get_completion_behavior<_Sndrs, _Env...>()...);
    }
  };

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {};
  }
};

template <class... _Sndrs>
_CCCL_NODEBUG_API constexpr auto when_all_t::operator()(_Sndrs... __sndrs) const
{
  if constexpr (sizeof...(_Sndrs) == 0)
  {
    return __sndr_t{};
  }
  else if constexpr (!__is_instantiable_with<::cuda::std::common_type_t, __early_domain_of_t<_Sndrs>...>)
  {
    static_assert(__is_instantiable_with<::cuda::std::common_type_t, __early_domain_of_t<_Sndrs>...>,
                  "when_all: all child senders must have the same domain");
  }
  else
  {
    using __dom_t _CCCL_NODEBUG_ALIAS = ::cuda::std::common_type_t<__early_domain_of_t<_Sndrs>...>;
    // If the incoming senders are non-dependent, we can check the completion
    // signatures of the composed sender immediately.
    if constexpr (((!dependent_sender<_Sndrs>) && ...))
    {
      __assert_valid_completion_signatures(get_completion_signatures<__sndr_t<_Sndrs...>>());
    }
    return transform_sender(__dom_t{}, __sndr_t<_Sndrs...>{{{}, {}, static_cast<_Sndrs&&>(__sndrs)...}});
  }
}

template <class... _Sndrs>
inline constexpr size_t structured_binding_size<when_all_t::__sndr_t<_Sndrs...>> = sizeof...(_Sndrs) + 2;

_CCCL_GLOBAL_CONSTANT when_all_t when_all{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_WHEN_ALL
